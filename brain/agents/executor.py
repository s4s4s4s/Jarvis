from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

from brain.client import chat, MODEL_HEAVY
from brain.agents.types import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-agent handlers
# Each handler receives (task: Task, context: dict[str, str]) -> str artifact
# context = {task_id: artifact} for all previously completed tasks
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
        {
            "role": "user",
            "content": f"{ctx_text}Research task: {task.goal}",
        },
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_code(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python developer. Write clean, working code. "
                "Return ONLY the code with minimal comments, no explanations outside code blocks."
            ),
        },
        {
            "role": "user",
            "content": f"{ctx_text}Code task: {task.goal}",
        },
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_audit(task: Task, context: dict[str, str]) -> str:
    """Calls AuditorAgent if code artifacts exist, otherwise falls back to LLM review."""
    # Collect code artifacts from dependencies
    code_artifacts = [
        context[dep] for dep in task.depends_on if dep in context
    ]

    if not code_artifacts:
        # No code artifacts — generic LLM review
        ctx_text = _build_context_block(task, context)
        messages = [
            {
                "role": "system",
                "content": "You are a code auditor. Find bugs, security issues, and inefficiencies.",
            },
            {
                "role": "user",
                "content": f"{ctx_text}Audit task: {task.goal}",
            },
        ]
        return chat(model=MODEL_HEAVY, messages=messages)

    # Write combined artifact to a temp file and run AuditorAgent
    try:
        from dev.auditor import AuditorAgent
        tmp_path = Path("logs/_executor_audit_tmp.py")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text("\n\n# --- next artifact ---\n\n".join(code_artifacts), encoding="utf-8")
        agent = AuditorAgent()
        findings = agent.audit([str(tmp_path)])
        if not findings:
            return "Audit complete: no issues found."
        lines = [f"[{f.type}] line {f.line} (conf={f.confidence:.2f}): {f.description} | Fix: {f.suggestion}"]
        return "Audit findings:\n" + "\n".join(
            f"  [{f.type}] line {f.line} conf={f.confidence:.2f}: {f.description} | Fix: {f.suggestion}"
            for f in findings
        )
    except Exception as e:
        logger.warning("[Executor] AuditorAgent failed (%s), falling back to LLM review", e)
        ctx_text = _build_context_block(task, context)
        messages = [
            {"role": "system", "content": "You are a code auditor. Find bugs and issues."},
            {"role": "user", "content": f"{ctx_text}Audit: {task.goal}"},
        ]
        return chat(model=MODEL_HEAVY, messages=messages)


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
        {
            "role": "user",
            "content": f"{ctx_text}Synthesis task: {task.goal}",
        },
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_chat(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide a clear, user-friendly final answer.",
        },
        {
            "role": "user",
            "content": f"{ctx_text}Task: {task.goal}",
        },
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


_AGENT_DISPATCH: dict[str, Callable[[Task, dict[str, str]], str]] = {
    "research": _run_research,
    "code": _run_code,
    "audit": _run_audit,
    "synthesize": _run_synthesize,
    "chat": _run_chat,
}


def _build_context_block(task: Task, context: dict[str, str]) -> str:
    """Build a context string from artifacts of dependency tasks."""
    if not task.depends_on:
        return ""
    parts = []
    for dep_id in task.depends_on:
        if dep_id in context:
            parts.append(f"=== Result of {dep_id} ===\n{context[dep_id]}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Sequential executor: runs tasks in order, respecting depends_on.
    Stops on first failed task.
    """

    def run(self, tasks: list[Task]) -> dict[str, str]:
        """
        Execute all tasks sequentially.
        Returns dict {task_id: artifact} for completed tasks.
        """
        context: dict[str, str] = {}  # task_id -> artifact

        for task in tasks:
            # Check all dependencies are done
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
# Runs full pipeline: PlannerAgent -> Executor
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

    # Step 1: Plan
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

    # Step 2: Execute
    executor = Executor()
    context = executor.run(plan)

    # Summary
    done = [t for t in plan if t.status == "done"]
    failed = [t for t in plan if t.status == "failed"]
    pending = [t for t in plan if t.status == "pending"]

    print(f"\n[Pipeline] Done: {len(done)}/{len(plan)} tasks")
    if failed:
        print(f"[Pipeline] Failed: {[t.id for t in failed]}")
    if pending:
        print(f"[Pipeline] Not reached: {[t.id for t in pending]}")

    # Print final artifact (last done task)
    if done:
        last = done[-1]
        print(f"\n[Pipeline] Final artifact from {last.id} [{last.type}]:")
        print("-" * 60)
        print(last.artifact)
        print("-" * 60)
