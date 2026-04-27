from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Callable

from brain.client import chat, MODEL_HEAVY
from brain.agents.types import Task

logger = logging.getLogger(__name__)

DEFAULT_CRITIC_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _findings_hash(report: str) -> str:
    return hashlib.md5(report.strip().encode()).hexdigest()


def _build_context_block(task: Task, context: dict[str, str], max_chars: int = 3000) -> str:
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


def _build_code_context(task: Task, context: dict[str, str], all_tasks: list[Task]) -> str:
    """
    Walk the dependency chain upward, collect the most recent actual CODE
    artifact (skipping audit stubs like 'Audit complete: no issues found.').
    """
    task_by_id = {t.id: t for t in all_tasks}
    visited: set[str] = set()
    code_parts: list[str] = []

    def _collect(dep_id: str) -> None:
        if dep_id in visited:
            return
        visited.add(dep_id)
        artifact = context.get(dep_id, "")
        dep_task = task_by_id.get(dep_id)
        if dep_task and dep_task.type == "code" and len(artifact.strip()) > 50:
            code_parts.append(f"# --- {dep_id} ---\n{artifact}")
        elif dep_task:
            for parent_dep in dep_task.depends_on:
                _collect(parent_dep)

    for dep_id in task.depends_on:
        _collect(dep_id)

    if not code_parts:
        return ""
    joined = "\n\n# ========================\n\n".join(code_parts)
    return f"=== Existing code to extend ===\n{joined}\n\n"


def _collect_code_artifacts(context: dict[str, str], tasks: list[Task]) -> str:
    code_task_ids = {t.id for t in tasks if t.type == "code"}
    parts: list[tuple[str, str]] = []
    for tid, art in context.items():
        base_id = tid.split("_fix")[0]
        if base_id in code_task_ids and len(art.strip()) > 50:
            parts.append((tid, art))

    if not parts:
        parts = [(tid, art) for tid, art in context.items()
                 if "def " in art or "import " in art]
    if not parts:
        return ""

    seen_bases: dict[str, tuple[str, str]] = {}
    for tid, art in parts:
        base = tid.split("_fix")[0]
        seen_bases[base] = (tid, art)

    return "\n\n# ========================\n\n".join(
        f"# --- {tid} ---\n{art}" for tid, art in seen_bases.values()
    )


def _strip_code_fences(text: str) -> str:
    """Extract raw code from ```python ... ``` block if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        return "\n".join(inner)
    return text


# ---------------------------------------------------------------------------
# Sub-agent handlers
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


def _run_code(
    task: Task,
    context: dict[str, str],
    fix_instructions: str | None = None,
    all_tasks: list[Task] | None = None,
) -> str:
    if all_tasks:
        ctx_text = _build_code_context(task, context, all_tasks)
    else:
        ctx_text = _build_context_block(task, context)

    if fix_instructions:
        user_content = (
            f"{ctx_text}Code task: {task.goal}\n\n"
            f"IMPORTANT — fix the following issues:\n{fix_instructions}"
        )
    else:
        user_content = f"{ctx_text}Code task: {task.goal}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python developer. Write clean, working code. "
                "If existing code is provided, EXTEND it — do not rewrite from scratch. "
                "Return ONLY the complete final code inside a ```python ... ``` block."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return chat(model=MODEL_HEAVY, messages=messages)


def _run_audit_raw(task: Task, code_artifact: str) -> tuple[str, bool]:
    """LLM-based audit fallback."""
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


def _run_audit(
    task: Task,
    context: dict[str, str],
    max_retries: int = DEFAULT_CRITIC_RETRIES,
    all_tasks: list[Task] | None = None,
) -> str:
    """
    Critic loop:
    1. Sandbox run — catch real runtime errors first
    2. LLM audit — catch logic/style issues
    3. If issues found → fix → repeat
    Stops on pass or stagnation.
    """
    from brain.tools.sandbox import sandbox_audit, has_infinite_loop

    if all_tasks:
        code_ctx = _build_code_context(task, context, all_tasks)
        code_artifact = code_ctx.replace("=== Existing code to extend ===\n", "").strip()
    else:
        code_artifacts = [context[dep] for dep in task.depends_on if dep in context]
        code_artifact = "\n\n".join(code_artifacts)

    if not code_artifact or len(code_artifact.strip()) < 20:
        ctx_text = _build_context_block(task, context)
        messages = [
            {"role": "system", "content": "You are a code auditor. Find bugs, security issues, and inefficiencies."},
            {"role": "user", "content": f"{ctx_text}Audit task: {task.goal}"},
        ]
        return chat(model=MODEL_HEAVY, messages=messages)

    current_code = _strip_code_fences(code_artifact)
    prev_hash: str | None = None
    report = ""

    for attempt in range(1, max_retries + 1):
        logger.info("[Critic] Audit attempt %d/%d for task %s", attempt, max_retries, task.id)

        # --- Step 1: Sandbox ---
        sandbox_report, sandbox_has_issues = sandbox_audit(current_code)
        logger.info("[Critic][Sandbox] %s", sandbox_report[:120])

        if sandbox_has_issues:
            combined_report = f"[Sandbox runtime error]:\n{sandbox_report}"
        else:
            # --- Step 2: LLM audit (only if sandbox passed) ---
            llm_report, llm_has_issues = _run_audit_raw(task, current_code)
            combined_report = llm_report
            sandbox_has_issues = llm_has_issues  # reuse flag

        has_issues = sandbox_has_issues

        if not has_issues:
            logger.info("[Critic] Task %s passed audit on attempt %d ✅", task.id, attempt)
            return combined_report

        current_hash = _findings_hash(combined_report)
        if current_hash == prev_hash:
            logger.warning("[Critic] Task %s: stagnation at attempt %d", task.id, attempt)
            return combined_report + f"\n\n[Critic] Stopped: stagnated at attempt {attempt}/{max_retries}."
        prev_hash = current_hash

        if attempt < max_retries:
            logger.info("[Critic] Issues found, requesting fix (attempt %d/%d)", attempt, max_retries)
            fix_task = Task(
                id=f"{task.id}_fix{attempt}",
                goal=task.goal,
                type="code",
                depends_on=task.depends_on,
                inputs=task.inputs,
            )
            fixed = _run_code(fix_task, context, fix_instructions=combined_report, all_tasks=all_tasks)
            current_code = _strip_code_fences(fixed)
            context[f"{task.id}_fix{attempt}"] = fixed
        else:
            logger.warning("[Critic] Task %s: max retries reached", task.id)

    return combined_report + f"\n\n[Critic] Max retries ({max_retries}) reached."


def _run_synthesize(
    task: Task,
    context: dict[str, str],
    all_tasks: list[Task] | None = None,
    user_request: str = "",
) -> str:
    from brain.tools.file_system import FileSystemTool, extract_code_blocks, suggest_filename

    all_code = _collect_code_artifacts(context, tasks=all_tasks or [])
    if not all_code:
        all_code = _build_context_block(task, context)
    if len(all_code) > 6000:
        all_code = all_code[:6000] + "\n... [truncated]"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a synthesis agent producing the FINAL deliverable.\n"
                "Rules:\n"
                "1. Combine provided code artifacts into ONE complete runnable Python script.\n"
                "2. Remove duplicate imports. Merge all functions/classes logically.\n"
                "3. Output ONLY the final script inside a ```python ... ``` code block.\n"
                "4. After the block: list pip deps and exact run command.\n"
                "5. Do NOT invent new functionality — only use what is in the artifacts."
            ),
        },
        {
            "role": "user",
            "content": f"User request: {user_request}\n\nCode artifacts:\n\n{all_code}",
        },
    ]
    llm_output = chat(model=MODEL_HEAVY, messages=messages)

    fs = FileSystemTool()
    blocks = extract_code_blocks(llm_output)
    saved_files: list[Path] = []

    name_source = user_request or task.goal
    for i, (lang, code) in enumerate(blocks):
        if lang in ("python", "py", "") and len(code.strip()) > 50:
            filename = suggest_filename(name_source, lang="py")
            if i > 0:
                filename = filename.replace(".py", f"_{i}.py")
            saved_path = fs.write_file(filename, code)
            saved_files.append(saved_path)
            logger.info("[Synthesize] Saved code to %s", saved_path)

    if saved_files:
        paths_str = "\n".join(f"  → {p.resolve()}" for p in saved_files)
        return f"[Synthesize] Done. Saved {len(saved_files)} file(s):\n{paths_str}\n\n" + llm_output
    else:
        logger.warning("[Synthesize] No code blocks found, returning raw output")
        return llm_output


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


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    def __init__(self, critic_retries: int = DEFAULT_CRITIC_RETRIES) -> None:
        self.critic_retries = critic_retries

    def run(self, tasks: list[Task], user_request: str = "") -> dict[str, str]:
        context: dict[str, str] = {}

        for task in tasks:
            for dep in task.depends_on:
                dep_task = next((t for t in tasks if t.id == dep), None)
                if dep_task is None:
                    logger.error("[Executor] Dependency '%s' not found", dep)
                    task.status = "failed"
                    return context
                if dep_task.status != "done":
                    logger.error("[Executor] Task %s depends on %s (status=%s)", task.id, dep, dep_task.status)
                    task.status = "failed"
                    return context

            logger.info("[Executor] Running %s [%s]: %s", task.id, task.type, task.goal)
            task.status = "running"
            try:
                if task.type == "audit":
                    artifact = _run_audit(task, context, max_retries=self.critic_retries, all_tasks=tasks)
                elif task.type == "synthesize":
                    artifact = _run_synthesize(task, context, all_tasks=tasks, user_request=user_request)
                elif task.type == "code":
                    artifact = _run_code(task, context, all_tasks=tasks)
                else:
                    handler: Callable[[Task, dict[str, str]], str] | None = {
                        "research": _run_research,
                        "chat": _run_chat,
                    }.get(task.type)
                    if handler is None:
                        logger.error("[Executor] Unknown task type '%s'", task.type)
                        task.status = "failed"
                        return context
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
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Jarvis agentic pipeline")
    parser.add_argument("request", nargs="+", help="User request")
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_CRITIC_RETRIES,
        help=f"Max Critic loop retries per audit task (default: {DEFAULT_CRITIC_RETRIES})",
    )
    args = parser.parse_args()

    from brain.agents.planner import PlannerAgent

    user_req = " ".join(args.request)
    print(f"\n[Pipeline] Request: {user_req}")
    print(f"[Pipeline] Critic retries: {args.retries}\n")

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

    executor = Executor(critic_retries=args.retries)
    context = executor.run(plan, user_request=user_req)

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
