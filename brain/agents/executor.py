"""brain/agents/executor.py

Asynchronous Executor with:
  1. Parallel wave  — independent tasks run concurrently
  2. Serial phase   — dependent tasks run sequentially
  3. CriticAgent    — every code task goes through review→fix loop
  4. ProjectContext — persistent state across conversations

Flow:
  Executor.run(tasks) →
      _run_parallel_wave(independent_tasks)   ← asyncio.gather
      _run_serial(dependent_tasks)            ← one by one
      Every code task: CriticAgent.review_and_fix() loop until green
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from brain.client import chat, chat_async, MODEL_FAST, MODEL_HEAVY, set_backend
from brain.agents.types import Task
from brain.agents.critic_agent import review_and_fix
from brain.agents.project_context import ProjectContext

logger = logging.getLogger(__name__)

DEFAULT_CRITIC_RETRIES = 3
JARVIS_ROOT = Path("C:/jarvis")

# Global project context — loaded once, shared across executor runs
_project_ctx: ProjectContext | None = None


def get_project_context() -> ProjectContext:
    global _project_ctx
    if _project_ctx is None:
        _project_ctx = ProjectContext.load()
    return _project_ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _findings_key(finding_line: str) -> tuple[str, str]:
    type_match = re.search(r"\[([A-Z_]+)\]", finding_line)
    line_match = re.search(r"line (\d+)", finding_line)
    ftype = type_match.group(1) if type_match else ""
    fline = line_match.group(1) if line_match else ""
    return (ftype, fline)


def _findings_normalized_set(report: str) -> frozenset[tuple[str, str]]:
    lines = [l.strip() for l in report.splitlines() if l.strip()]
    return frozenset(_findings_key(l) for l in lines if "[" in l and "line" in l)


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


def _fix_pip_deps_in_output(llm_output: str, final_code: str) -> str:
    from brain.tools.sandbox import extract_pip_requirements
    real_deps = extract_pip_requirements(final_code)

    if not real_deps:
        replacement = "(no third-party pip dependencies)"
    else:
        replacement = " ".join(real_deps)

    llm_output = re.sub(
        r"(```bash\s*pip install\s*)[^`]+(```)",
        lambda m: f"{m.group(1)}{replacement}{m.group(2)}",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    llm_output = re.sub(
        r"pip install[^\n]+",
        f"pip install {replacement}",
        llm_output,
        flags=re.IGNORECASE,
    )
    return llm_output


def _git_stage_and_commit(saved_files: list[Path], goal: str) -> str | None:
    """Commit generated files to an ISOLATED local branch."""
    branch_name = f"jarvis/synthesize-{int(time.time())}"
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=str(JARVIS_ROOT), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "add"] + [str(p.resolve()) for p in saved_files],
            cwd=str(JARVIS_ROOT), check=True, capture_output=True,
        )
        msg = f"jarvis/synthesize: {goal[:72]}"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(JARVIS_ROOT), check=True, capture_output=True,
        )
        logger.info("[Synthesize] Committed to isolated branch '%s'", branch_name)
        return branch_name
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "[Synthesize] git operation failed (non-fatal): %s\nstderr: %s",
            exc, exc.stderr.decode(errors="replace") if exc.stderr else "",
        )
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=str(JARVIS_ROOT), check=False, capture_output=True,
        )
        return None
    except Exception as exc:
        logger.warning("[Synthesize] git commit failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def _run_tool(task: Task, context: dict[str, str]) -> str:
    from tools.registry import call_tool
    result = call_tool(task.tool_name, task.inputs)
    if result.ok:
        data = result.data
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)
    return f"[Tool error: {task.tool_name}] {result.error}"


async def _run_tool_async(task: Task, context: dict[str, str]) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _run_tool(task, context))


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
    """
    Write code AND immediately pass it through CriticAgent.
    The critic runs up to MAX_CRITIC_ROUNDS of review→fix iterations.
    Returns the best version of the code (critic-approved if possible).
    """
    if all_tasks:
        ctx_text = _build_code_context(task, context, all_tasks)
    else:
        ctx_text = _build_context_block(task, context)

    # Inject project context so the author knows what already exists
    proj_ctx = get_project_context()
    proj_summary = proj_ctx.summary() if not proj_ctx.is_empty() else ""

    if fix_instructions:
        user_content = (
            f"{ctx_text}Code task: {task.goal}\n\n"
            f"IMPORTANT — fix the following issues:\n{fix_instructions}"
        )
    else:
        user_content = f"{ctx_text}Code task: {task.goal}"

    if proj_summary:
        user_content = f"PROJECT CONTEXT:\n{proj_summary}\n\n" + user_content

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python developer. Write clean, production-ready code. "
                "If existing code is provided, EXTEND it — do not rewrite from scratch. "
                "Handle edge cases. Include proper error handling. "
                "Return ONLY the complete final code inside a ```python ... ``` block."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    raw = chat(model=MODEL_HEAVY, messages=messages)
    initial_code = _strip_code_fences(raw)

    # ----------------------------------------------------------------
    # CriticAgent loop — review → fix until no blocking issues
    # ----------------------------------------------------------------
    from brain.ask import report_progress
    report_progress(f"🔍 Critic reviewing: {task.goal[:60]}...")

    critic_result = review_and_fix(initial_code, task.goal)

    logger.info(
        "[Executor] Code task %s — %s",
        task.id, critic_result.summary(),
    )
    report_progress(f"{'✅' if critic_result.passed else '⚠️'} {critic_result.summary()}")

    final_code = critic_result.fixed_code or initial_code

    # Update project context with this file
    if final_code:
        proj_ctx.add_file(
            rel_path=f"output/{task.id}.py",
            code=final_code,
            summary=task.goal[:80],
        )
        proj_ctx.record_test(
            name=f"critic_{task.id}",
            passed=critic_result.passed,
            details=critic_result.summary(),
        )
        if not critic_result.passed:
            for issue in critic_result.blocking_issues()[:3]:
                proj_ctx.record_error(f"{task.id}: [{issue.severity}] {issue.description}")
        proj_ctx.save()

    return final_code


def _run_audit_raw(task: Task, code_artifact: str) -> tuple[str, bool]:
    Path("logs").mkdir(parents=True, exist_ok=True)
    try:
        from dev.auditor import AuditorAgent, GENERIC_SYSTEM_PROMPT
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="jarvis_audit_",
            dir="logs", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code_artifact)
            tmp_path = Path(tmp.name)
        try:
            agent = AuditorAgent(system_prompt=GENERIC_SYSTEM_PROMPT)
            findings = agent.audit([str(tmp_path)])
        finally:
            tmp_path.unlink(missing_ok=True)
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
            {"role": "system", "content": "You are a code auditor. Find bugs, security issues, and inefficiencies. Be concise."},
            {"role": "user", "content": f"Audit this code:\n\n{code_artifact[:3000]}\n\nTask: {task.goal}"},
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
    from brain.tools.sandbox import sandbox_audit

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
    prev_findings: frozenset[tuple[str, str]] | None = None
    combined_report = ""

    for attempt in range(1, max_retries + 1):
        logger.info("[Critic] Audit attempt %d/%d for task %s", attempt, max_retries, task.id)

        sandbox_report, sandbox_has_issues = sandbox_audit(current_code)
        logger.info("[Critic][Sandbox] %s", sandbox_report[:120])

        if sandbox_has_issues:
            combined_report = f"[Sandbox runtime error]:\n{sandbox_report}"
        else:
            llm_report, llm_has_issues = _run_audit_raw(task, current_code)
            combined_report = llm_report
            sandbox_has_issues = llm_has_issues

        has_issues = sandbox_has_issues

        if not has_issues:
            logger.info("[Critic] Task %s passed audit on attempt %d ✅", task.id, attempt)
            return combined_report

        current_findings = _findings_normalized_set(combined_report)
        if prev_findings is not None and current_findings == prev_findings:
            logger.warning("[Critic] Task %s: stagnation at attempt %d", task.id, attempt)
            return combined_report + f"\n\n[Critic] Stopped: stagnated at attempt {attempt}/{max_retries}."
        prev_findings = current_findings

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
            current_code = fixed
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
    final_code = ""

    name_source = user_request or task.goal
    for i, (lang, code) in enumerate(blocks):
        if lang in ("python", "py", "") and len(code.strip()) > 50:
            final_code = code
            filename = suggest_filename(name_source, lang="py")
            if i > 0:
                filename = filename.replace(".py", f"_{i}.py")
            saved_path = fs.write_file(filename, code)
            saved_files.append(saved_path)
            logger.info("[Synthesize] Saved code to %s", saved_path)

            # Register in project context
            ctx = get_project_context()
            ctx.add_file(str(saved_path), code, summary=task.goal[:80])
            ctx.close_task(task.goal)
            ctx.save()

    if final_code:
        llm_output = _fix_pip_deps_in_output(llm_output, final_code)

    if saved_files:
        branch = _git_stage_and_commit(saved_files, task.goal)
        paths_str = "\n".join(f"  → {p.resolve()}" for p in saved_files)
        branch_note = (
            f"\n⚠️  Код сохранён в изолированную ветку: {branch}\n"
            f"    Проверь и смержи вручную: git diff feature/planner-agent..{branch}"
            if branch else
            "\n⚠️  git commit не удался — файлы сохранены на диск, но не закоммичены."
        )
        return (
            f"[Synthesize] Done. Saved {len(saved_files)} file(s):\n{paths_str}"
            + branch_note
            + "\n\n" + llm_output
        )
    else:
        logger.warning("[Synthesize] No code blocks found, returning raw output")
        return llm_output


def _run_chat(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide a clear, user-friendly final answer."},
        {"role": "user", "content": f"{ctx_text}Task: {task.goal}"},
    ]
    return chat(model=MODEL_FAST, messages=messages)


# ---------------------------------------------------------------------------
# Async sub-agent handlers
# ---------------------------------------------------------------------------

async def _run_research_async(task: Task, context: dict[str, str]) -> str:
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
    return await chat_async(model=MODEL_HEAVY, messages=messages)


async def _run_code_async(
    task: Task,
    context: dict[str, str],
    all_tasks: list[Task] | None = None,
) -> str:
    # Async wrapper — runs sync _run_code (which includes CriticAgent) in executor
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _run_code(task, context, all_tasks=all_tasks)
    )


async def _run_chat_async(task: Task, context: dict[str, str]) -> str:
    ctx_text = _build_context_block(task, context)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide a clear, user-friendly final answer."},
        {"role": "user", "content": f"{ctx_text}Task: {task.goal}"},
    ]
    return await chat_async(model=MODEL_FAST, messages=messages)


async def _run_task_async(
    task: Task,
    context: dict[str, str],
    all_tasks: list[Task],
) -> tuple[str, str]:
    logger.info("[Executor][parallel] %s [%s]: %s", task.id, task.type, task.goal)
    task.status = "running"
    if task.type == "research":
        artifact = await _run_research_async(task, context)
    elif task.type == "code":
        artifact = await _run_code_async(task, context, all_tasks)
    elif task.type == "chat":
        artifact = await _run_chat_async(task, context)
    elif task.type == "tool":
        artifact = await _run_tool_async(task, context)
    else:
        loop = asyncio.get_running_loop()
        if task.type == "audit":
            artifact = await loop.run_in_executor(
                None, lambda: _run_audit(task, context, all_tasks=all_tasks)
            )
        elif task.type == "synthesize":
            artifact = await loop.run_in_executor(
                None, lambda: _run_synthesize(task, context, all_tasks=all_tasks)
            )
        else:
            logger.error("[Executor][parallel] Unknown task type '%s'", task.type)
            task.status = "failed"
            return task.id, ""
    task.artifact = artifact
    task.status = "done"
    return task.id, artifact


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    def __init__(
        self,
        critic_retries: int = DEFAULT_CRITIC_RETRIES,
        use_parallel: bool = True,
    ) -> None:
        self.critic_retries = critic_retries
        self.use_parallel   = use_parallel

    def run(self, tasks: list[Task], user_request: str = "") -> dict[str, str]:
        # Register open tasks in project context
        ctx = get_project_context()
        for t in tasks:
            if t.type == "code":
                ctx.open_task(t.goal)
        ctx.save()

        if self.use_parallel:
            independent = [t for t in tasks if not t.depends_on]
            if len(independent) > 1:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            asyncio.run,
                            self._run_with_parallel(tasks, user_request)
                        )
                        return future.result()
                return asyncio.run(self._run_with_parallel(tasks, user_request))
        return self._run_serial(tasks, user_request)

    async def _run_with_parallel(
        self, tasks: list[Task], user_request: str
    ) -> dict[str, str]:
        context: dict[str, str] = {}
        independent = [t for t in tasks if not t.depends_on]
        dependent   = [t for t in tasks if t.depends_on]

        tool_only_wave = all(t.type == "tool" for t in independent)

        if tool_only_wave:
            logger.info(
                "[Executor] Parallel wave: %d tool tasks → skipping llama-server",
                len(independent),
            )
            results = await asyncio.gather(
                *[_run_task_async(t, context, tasks) for t in independent],
                return_exceptions=True,
            )
        else:
            from brain.llama_server import LlamaServerManager
            logger.info(
                "[Executor] Parallel wave: %d tasks (LLM) → launching llama-server",
                len(independent),
            )
            try:
                async with LlamaServerManager() as srv:
                    set_backend("llama")
                    logger.info("[Executor] llama-server ready at %s", srv.base_url)
                    results = await asyncio.gather(
                        *[_run_task_async(t, context, tasks) for t in independent],
                        return_exceptions=True,
                    )
            finally:
                set_backend("ollama")

        for res in results:
            if isinstance(res, Exception):
                logger.error("[Executor][parallel] Task raised: %s", res)
                continue
            task_id, artifact = res
            context[task_id] = artifact
            for t in independent:
                if t.id == task_id:
                    t.status = "done" if artifact else "failed"

        if dependent:
            logger.info(
                "[Executor] Serial phase: %d dependent tasks (Ollama)", len(dependent)
            )
            serial_context = self._run_serial(
                dependent, user_request, context=context, all_tasks=tasks
            )
            context.update(serial_context)

        return context

    def _run_serial(
        self,
        tasks: list[Task],
        user_request: str = "",
        context: dict[str, str] | None = None,
        all_tasks: list[Task] | None = None,
    ) -> dict[str, str]:
        if context is None:
            context = {}
        effective_all = all_tasks or tasks
        errors: list[str] = []

        for task in tasks:
            dep_failed = False
            for dep in task.depends_on:
                dep_task = next((t for t in effective_all if t.id == dep), None)
                if dep_task is None:
                    logger.error("[Executor] Dependency '%s' not found for task '%s'", dep, task.id)
                    dep_failed = True
                    break
                if dep_task.status != "done":
                    logger.error(
                        "[Executor] Task %s depends on %s (status=%s) — skipping",
                        task.id, dep, dep_task.status,
                    )
                    dep_failed = True
                    break
            if dep_failed:
                task.status = "failed"
                errors.append(task.id)
                continue

            logger.info("[Executor] Running %s [%s]: %s", task.id, task.type, task.goal)
            task.status = "running"
            try:
                if task.type == "audit":
                    artifact = _run_audit(
                        task, context,
                        max_retries=self.critic_retries,
                        all_tasks=effective_all,
                    )
                elif task.type == "synthesize":
                    artifact = _run_synthesize(
                        task, context,
                        all_tasks=effective_all,
                        user_request=user_request,
                    )
                elif task.type == "code":
                    artifact = _run_code(task, context, all_tasks=effective_all)
                elif task.type == "tool":
                    artifact = _run_tool(task, context)
                else:
                    handler: Callable[[Task, dict[str, str]], str] | None = {
                        "research": _run_research,
                        "chat":     _run_chat,
                    }.get(task.type)
                    if handler is None:
                        logger.error("[Executor] Unknown task type '%s'", task.type)
                        task.status = "failed"
                        errors.append(task.id)
                        continue
                    artifact = handler(task, context)

                task.artifact = artifact
                task.status = "done"
                context[task.id] = artifact
                logger.info("[Executor] %s done (%d chars)", task.id, len(artifact))
            except Exception as e:
                logger.error("[Executor] Task %s failed: %s", task.id, e)
                task.status = "failed"
                errors.append(task.id)
                # Record error in project context
                get_project_context().record_error(f"{task.id}: {e}")
                get_project_context().save()

        if errors:
            logger.warning("[Executor] Pipeline completed with %d failed task(s): %s", len(errors), errors)
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
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel execution (use Ollama serial mode)",
    )
    args = parser.parse_args()

    from brain.agents.planner import PlannerAgent

    user_req = " ".join(args.request)
    print(f"\n[Pipeline] Request: {user_req}")
    print(f"[Pipeline] Critic retries: {args.retries}")
    print(f"[Pipeline] Parallel: {not args.no_parallel}\n")

    planner = PlannerAgent()
    try:
        plan = planner.plan(user_req)
    except Exception as e:
        print(f"[Pipeline] Planner ERROR: {e}")
        sys.exit(1)

    print(f"[Pipeline] Plan: {len(plan)} tasks")
    for t in plan:
        deps = f" <- {t.depends_on}" if t.depends_on else ""
        tool = f" [tool={t.tool_name}]" if t.tool_name else ""
        print(f"  {t.id} [{t.type}]{tool} {t.goal}{deps}")
    print()

    executor = Executor(critic_retries=args.retries, use_parallel=not args.no_parallel)
    context = executor.run(plan, user_request=user_req)

    done    = [t for t in plan if t.status == "done"]
    failed  = [t for t in plan if t.status == "failed"]
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
