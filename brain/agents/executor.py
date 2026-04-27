from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

from brain.client import chat, MODEL_HEAVY
from brain.agents.types import Task

logger = logging.getLogger(__name__)

CRITIC_MAX_RETRIES = 3  # max fix iterations per audit task


# ---------------------------------------------------------------------------
# Sub-agent handlers
# Each handler receives (task: Task, context: dict[str, str]) -> str artifact
# ---------------------------------------------------------------------------

def _run_research(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research agent. Answer thoroughly and factually. "
                "Return a structured markdown report."
            ),
        },
        {"role": "user", "content": f"{ctx_text}Research task: {task.goal}"},
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_code(task: Task, context: dict[str, str], fix_instructions: str | None = None) -> str:
    ctx_text = _build_context_block(task, context)
    if fix_instructions:
        user_content = (
            f"{ctx_text}Code task: {task.goal}\n\n"
            f"IMPORTANT — fix the following issues found during audit:\n{fix_instructions}"
        )
    else:
        user_content = f"{ctx_text}Code task: {task.goal}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python developer. Write clean, working code. "
                "Return ONLY the code with minimal comments, no explanations outside code blocks."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_audit_raw(task: Task, code_artifact: str) -> tuple[str, bool]:
    """
    Run AuditorAgent with generic prompt on code_artifact.
    Returns (audit_report: str, has_issues: bool).
    """
    try:
        from dev.auditor import AuditorAgent, GENERIC_SYSTEM_PROMPT
        tmp_path = Path("logs/_executor_audit_tmp.py")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(code_artifact, encoding="utf-8")
        agent = AuditorAgent(system_prompt=GENERIC_SYSTEM_PROMPT)
        findings = agent.audit([str(tmp_path)])
        confirmed = [f for f in findings if f.status == "confirmed"]
        if not confirmed:
            return "Audit complete: no issues found.", False
        report = "Audit findings:\n" + "\n".join(
            f"  [{f.type}] line {f.line} conf={f.confidence:.2f}: {f.description} | Fix: {f.suggestion}"
            for f in confirmed
        )
        return report, True
    except Exception as e:
        logger.warning("[Executor] AuditorAgent failed (%s), falling back to LLM audit", e)
        messages = [
            {
                "role": "system",
                "content": "You are a code auditor. Find bugs, security issues, and inefficiencies. Be concise.",
            },
            {
                "role": "user",
                "content": f"Audit this code:\n\n{code_artifact[:3000]}\n\nTask: {task.goal}",
            },
        ]
        report = chat(model=MODEL_HEAVY, messages=messages)
        has_issues = "no issues" not in report.lower() and len(report.strip()) > 20
        return report, has_issues


def _run_audit(task: Task, context: dict[str, str]) -> str:
    """
    Critic loop: audit → if issues → fix code → audit again. Max CRITIC_MAX_RETRIES iterations.
    Returns final audit report.
    """
    # Collect code from direct dependencies
    code_artifacts = [
        context[dep] for dep in task.depends_on if dep in context
    ]

    if not code_artifacts:
        # Nothing to audit — generic LLM review
        ctx_text = _build_context_block(task, context)
        messages = [
            {
                "role": "system",
                "content": "You are a code auditor. Find bugs, security issues, and inefficiencies.",
            },
            {"role": "user", "content": f"{ctx_text}Audit task: {task.goal}"},
        ]
        return chat(model=MODEL_HEAVY, messages=messages)

    current_code = "\n\n# --- next artifact ---\n\n".join(code_artifacts)

    for attempt in range(1, CRITIC_MAX_RETRIES + 1):
        logger.info("[Critic] Audit attempt %d/%d for task %s", attempt, CRITIC_MAX_RETRIES, task.id)
        report, has_issues = _run_audit_raw(task, current_code)

        if not has_issues:
            logger.info("[Critic] Task %s passed audit on attempt %d", task.id, attempt)
            return report

        if attempt < CRITIC_MAX_RETRIES:
            logger.info("[Critic] Issues found, requesting fix (attempt %d)", attempt)
            # Re-run code with fix instructions, using same task but injecting findings
            fix_task = Task(
                id=f"{task.id}_fix{attempt}",
                goal=task.goal,
                type="code",
                depends_on=task.depends_on,
                inputs=task.inputs,
            )
            current_code = _run_code(fix_task, context, fix_instructions=report)
            # Update context so next audit sees fixed code
            context[f"{task.id}_fix{attempt}"] = current_code
        else:
            logger.warning(
                "[Critic] Task %s still has issues after %d attempts, continuing with last version",
                task.id, CRITIC_MAX_RETRIES,
            )

    return report + f"\n\n[Critic] Max retries ({CRITIC_MAX_RETRIES}) reached. Last code version kept."


def _run_synthesize(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a synthesis agent. Combine the provided artifacts into a "
                "coherent final result. Be concise and structured."
            ),
        },
        {"role": "user", "content": f"{ctx_text}Synthesis task: {task.goal}"},
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_chat(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide a clear, user-friendly final answer.",
        },
        {"role": "user", "content": f"{ctx_text}Task: {task.goal}"},
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


_AGENT_DISPATCH: dict[str, Callable[[Task, dict[str, str]], str]] = {
    "research": _run_research,
    "code": _run_code,
    "audit": _run_audit,
    "synthesize": _run_synthesize,
    "chat": _run_chat,
}


def _build_context_block(task: Task, context: dict[str, str], max_chars: int = 3000) -> str:
    """Build context string from direct dependency artifacts, truncated to max_chars each."""
    if not task.depends_on:
        return ""
    parts = []
    for dep_id in task.depends_on:
        if dep_id in context:
            artifact = context[dep_id]
            if len(artifact) > max_chars:
                artifact = artifact[:max_chars] + "\n... [truncated]"
            parts.append(f"=== Result of {dep_id} ===\n{artifact}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Sequential executor: runs tasks in order, respecting depends_on.
    Audit tasks use Critic loop (auto-fix up to CRITIC_MAX_RETRIES times).
    Stops on first failed task.
    """

    def run(self, tasks: list[Task]) -> dict[str, str]:
        context: dict[str, str] = {}

        for task in tasks:
            for dep in task.depends_on:
                dep_task = next((t for t in tasks if t.id == dep), None)
                if dep_task is None:
                    logger.error("[Executor] Dependency '%s' not found in plan", dep)
                    task.status = "failed"
                    return context
                if dep_task.status != "done":
                    logger.error(
                        "[Executor] Task %s depends on %s which is not done (status=%s)",
                        task.id, dep, dep_task.status,
                    )
                    task.status = "failed"
                    return context

            handler = _AGENT_DISPATCH.get(task.type)
            if handler is None:
                logger.error("[Executor] Unknown task type '%s' for task %s", task.type, task.id)
                task.status = "failed"
                return context

            logger.info("[Executor] Running %s [%s]: %s", task.id, task.type, task.goal)
            task.status = "running"
            try:
                artifact = handler(task, context)
                task.artifact = artifact
                task.status = "done"
                context[task.id] = artifact
                logger.info("[Executor] %s done (%d chars)", task.id, len(artifact))
            except Exception as e:
                logger.error("[Executor] Task %s failed: %s", task.id, e)
                task.status = "failed"
                return context

        return context


# ---------------------------------------------------------------------------
# CLI: python -m brain.agents.executor "<user request>"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print('Usage: python -m brain.agents.executor "<user request>"')
        sys.exit(1)

    from brain.agents.planner import PlannerAgent

    user_req = " ".join(sys.argv[1:])
    print(f"\n[Pipeline] Request: {user_req}\n")

    planner = PlannerAgent()
    try:
        plan = planner.plan(user_req)
    except Exception as e:
        print(f"[Pipeline] Planner ERROR: {e}")
        sys.exit(1)

    print(f"[Pipeline] Plan: {len(plan)} tasks")
    for t in plan:
        deps = f" <- {t.depends_on}" if t.depends_on else ""
        print(f"  {t.id} [{t.type}] {t.goal}{deps}")
    print()

    executor = Executor()
    context = executor.run(plan)

    done = [t for t in plan if t.status == "done"]
    failed = [t for t in plan if t.status == "failed"]
    pending = [t for t in plan if t.status == "pending"]

    print(f"\n[Pipeline] Done: {len(done)}/{len(plan)} tasks")
    if failed:
        print(f"[Pipeline] Failed: {[t.id for t in failed]}")
    if pending:
        print(f"[Pipeline] Not reached: {[t.id for t in pending]}")

    if done:
        last = done[-1]
        print(f"\n[Pipeline] Final artifact from {last.id} [{last.type}]:")
        print("-" * 60)
        print(last.artifact)
        print("-" * 60)
