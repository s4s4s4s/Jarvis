"""brain/agents/self_analysis_agent.py

SelfAnalysisAgent — Jarvis reads and understands his own source code.

Capabilities:
  1. scan()        — walk all .py files, build file registry with AST metadata
  2. analyse()     — LLM reads real source, finds concrete issues with file+line
  3. propose()     — for each issue: generate a diff-style patch
  4. apply()       — write patches to disk, record in ProjectContext
  5. run(query)    — public entry point called from _dispatch

Output format (to user):
  - Architecture overview
  - Issues list: [SEVERITY] file:line — description
  - Proposed patches per issue
  - Applied/skipped summary

This agent does NOT guess — it reads actual file contents before analysis.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.client import chat, MODEL_HEAVY
from brain.ask import report_progress

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files/dirs to skip during scan
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}
_SKIP_FILES = {"self_analysis_agent.py"}  # don't analyse ourselves recursively

# Max chars per file sent to LLM (truncate large files)
_MAX_FILE_CHARS = 8000
# Max files to send in one LLM call
_MAX_FILES_PER_BATCH = 6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    rel_path: str
    content: str
    size: int
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    syntax_ok: bool = True
    syntax_error: str = ""


@dataclass
class AnalysisIssue:
    severity: str        # CRITICAL / HIGH / MEDIUM / LOW
    file: str
    line: int
    description: str
    suggestion: str
    patch: str = ""      # filled by propose()

    @property
    def is_blocking(self) -> bool:
        return self.severity in ("CRITICAL", "HIGH")


@dataclass
class AnalysisResult:
    overview: str
    issues: list[AnalysisIssue] = field(default_factory=list)
    applied_patches: list[str] = field(default_factory=list)
    skipped_patches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.issues)
        blocking = sum(1 for i in self.issues if i.is_blocking)
        applied = len(self.applied_patches)
        lines = [f"🔍 Найдено проблем: {total} (блокирующих: {blocking})"]
        if applied:
            lines.append(f"✅ Применено патчей: {applied}")
        if self.skipped_patches:
            lines.append(f"⚠️ Пропущено (требует ручной проверки): {len(self.skipped_patches)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: File scanner
# ---------------------------------------------------------------------------

def _extract_ast_info(source: str) -> tuple[list[str], list[str], list[str], bool, str]:
    """Parse source with AST, extract functions/classes/imports."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [], [], [], False, str(e)

    functions, classes, imports = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return functions, classes, list(set(imports)), True, ""


def scan_repo(target_dirs: list[str] | None = None) -> list[FileInfo]:
    """
    Walk the repo and collect FileInfo for all .py files.
    If target_dirs given, only scan those (relative to REPO_ROOT).
    """
    files: list[FileInfo] = []

    if target_dirs:
        roots = [REPO_ROOT / d for d in target_dirs if (REPO_ROOT / d).exists()]
    else:
        # Default: scan brain/, tools/, core/ — skip output/, logs/, etc.
        roots = [
            REPO_ROOT / "brain",
            REPO_ROOT / "tools",
            REPO_ROOT / "core",
            REPO_ROOT / "dev",
        ]
        roots = [r for r in roots if r.exists()]

    for root in roots:
        for py_file in sorted(root.rglob("*.py")):
            # Skip excluded dirs
            if any(part in _SKIP_DIRS for part in py_file.parts):
                continue
            if py_file.name in _SKIP_FILES:
                continue
            try:
                rel = py_file.relative_to(REPO_ROOT).as_posix()
                content = py_file.read_text(encoding="utf-8", errors="replace")
                funcs, classes, imports, ok, err = _extract_ast_info(content)
                files.append(FileInfo(
                    rel_path=rel,
                    content=content,
                    size=len(content),
                    functions=funcs,
                    classes=classes,
                    imports=imports,
                    syntax_ok=ok,
                    syntax_error=err,
                ))
            except Exception as e:
                logger.warning("[SelfAnalysis] Failed to read %s: %s", py_file, e)

    logger.info("[SelfAnalysis] Scanned %d files", len(files))
    return files


# ---------------------------------------------------------------------------
# Step 2: Architecture overview
# ---------------------------------------------------------------------------

_OVERVIEW_SYSTEM = """\
You are a senior software architect doing a deep review of the Jarvis AI assistant codebase.

You receive a registry of Python files with their functions, classes, and import graph.
Produce a concise architecture overview in Russian:
  1. Core data flow (how a user request flows through the system)
  2. Key components and their responsibilities
  3. Inter-module dependencies (which files call which)
  4. Obvious architectural weaknesses or coupling issues

Be specific — mention actual file names and function names.
Max 400 words.
"""


def _build_registry_summary(files: list[FileInfo]) -> str:
    """Build a compact registry string for the LLM overview prompt."""
    lines = []
    for f in files:
        syntax = "OK" if f.syntax_ok else f"SyntaxError: {f.syntax_error}"
        funcs = ", ".join(f.functions[:10]) or "—"
        classes = ", ".join(f.classes[:5]) or "—"
        deps = ", ".join(f.imports[:8]) or "—"
        lines.append(
            f"{f.rel_path} [{syntax}] ({f.size} chars)\n"
            f"  functions: {funcs}\n"
            f"  classes:   {classes}\n"
            f"  imports:   {deps}"
        )
    return "\n\n".join(lines)


def build_overview(files: list[FileInfo]) -> str:
    registry = _build_registry_summary(files)
    messages = [
        {"role": "system", "content": _OVERVIEW_SYSTEM},
        {"role": "user", "content": f"File registry:\n\n{registry}"},
    ]
    return chat(MODEL_HEAVY, messages, options={"temperature": 0.1, "num_ctx": 16384})


# ---------------------------------------------------------------------------
# Step 3: Deep issue analysis (reads actual source)
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM = """\
You are a senior Python engineer doing a ruthless code review of the Jarvis AI assistant.

You receive the ACTUAL SOURCE CODE of multiple files.
Find every real issue:
  1. Bugs that will cause runtime errors or wrong behaviour
  2. Logic errors — wrong conditions, missing cases, broken flows
  3. Architectural problems — circular imports, missing error handling, silent failures
  4. Missing features that are referenced but not implemented
  5. Integration gaps — two components that should talk but don't
  6. Dead code or unreachable branches
  7. Hardcoded values that should be configurable
  8. Anything that would make this system fail in production

For each issue output ONE line in this exact format:
  [SEVERITY] file.py:LINE — description — fix: concrete fix instruction

SEVERITY: CRITICAL / HIGH / MEDIUM / LOW
LINE: exact line number (use 0 if file-level issue)

If no real issues — output: LGTM

Do NOT output prose. Do NOT output markdown. Only issue lines or LGTM.
"""


def _parse_issues(raw: str, files: list[FileInfo]) -> list[AnalysisIssue]:
    """Parse LLM issue output into AnalysisIssue objects."""
    known_files = {f.rel_path for f in files}
    short_to_full = {Path(f.rel_path).name: f.rel_path for f in files}

    issues: list[AnalysisIssue] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "LGTM":
            continue
        m = re.match(
            r"\[(CRITICAL|HIGH|MEDIUM|LOW)\]\s+"
            r"([\w./\-]+\.py)(?::(\d+))?\s*[\u2014\-]+\s*"
            r"(.+?)(?:\s*[\u2014\-]+\s*fix:\s*(.+))?",
            line, re.IGNORECASE,
        )
        if m:
            severity = m.group(1).upper()
            raw_file = m.group(2)
            line_num = int(m.group(3)) if m.group(3) else 0
            description = m.group(4).strip()
            fix = m.group(5).strip() if m.group(5) else "see description"

            if raw_file in known_files:
                resolved = raw_file
            elif raw_file in short_to_full:
                resolved = short_to_full[raw_file]
            else:
                matches = [f for f in known_files if f.endswith(raw_file)]
                resolved = matches[0] if matches else raw_file

            issues.append(AnalysisIssue(
                severity=severity,
                file=resolved,
                line=line_num,
                description=description,
                suggestion=fix,
            ))
        elif len(line) > 15 and not line.startswith("#"):
            issues.append(AnalysisIssue(
                severity="MEDIUM",
                file="unknown",
                line=0,
                description=line[:200],
                suggestion="manual review required",
            ))
    return issues


def analyse_files(files: list[FileInfo]) -> list[AnalysisIssue]:
    """
    Send files to LLM in batches, collect all issues.
    Batches by _MAX_FILES_PER_BATCH to stay within context window.
    """
    all_issues: list[AnalysisIssue] = []
    batches = [files[i:i+_MAX_FILES_PER_BATCH] for i in range(0, len(files), _MAX_FILES_PER_BATCH)]

    for batch_idx, batch in enumerate(batches):
        report_progress(f"🔍 Анализирую батч {batch_idx+1}/{len(batches)} ({len(batch)} файлов)...")

        parts = []
        for fi in batch:
            content = fi.content
            if len(content) > _MAX_FILE_CHARS:
                content = content[:_MAX_FILE_CHARS] + f"\n\n... [TRUNCATED at {_MAX_FILE_CHARS} chars]"
            parts.append(f"=== {fi.rel_path} ===\n{content}")
        source_dump = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": _ANALYSIS_SYSTEM},
            {"role": "user", "content": f"Review these files:\n\n{source_dump}"},
        ]
        try:
            raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.05, "num_ctx": 32768})
            batch_issues = _parse_issues(raw, batch)
            logger.info("[SelfAnalysis] Batch %d: %d issues found", batch_idx+1, len(batch_issues))
            all_issues.extend(batch_issues)
        except Exception as e:
            logger.error("[SelfAnalysis] Batch %d analysis failed: %s", batch_idx+1, e)

    return all_issues


# ---------------------------------------------------------------------------
# Step 4: Patch proposal
# ---------------------------------------------------------------------------

_PATCHER_SYSTEM = """\
You are a senior Python engineer generating a minimal, precise code patch.

You receive:
  1. The issue description and fix instruction
  2. The full source of the file to patch

Generate the MINIMAL change needed to fix this specific issue.
Output a JSON object:
  {
    "old_code": "exact lines to replace (5-15 lines of context)",
    "new_code": "replacement lines with the fix applied",
    "explanation": "one sentence what changed and why"
  }

Rules:
  - old_code must be an EXACT substring of the source file (copy-paste, no paraphrasing)
  - new_code must be syntactically valid Python
  - Change ONLY what is needed to fix the issue
  - If the fix requires adding a new function, include surrounding context in old_code
  - If you cannot generate a safe patch, set old_code and new_code to empty strings
  - Return ONLY valid JSON, no markdown
"""


def _generate_patch(issue: AnalysisIssue, source: str) -> dict[str, str]:
    """Ask LLM to generate old_code/new_code patch for one issue."""
    source_lines = source.splitlines()
    start = max(0, issue.line - 10)
    end = min(len(source_lines), issue.line + 20)
    context_snippet = "\n".join(
        f"{i+1}: {l}" for i, l in enumerate(source_lines[start:end], start=start)
    )

    messages = [
        {"role": "system", "content": _PATCHER_SYSTEM},
        {"role": "user", "content": (
            f"Issue: [{issue.severity}] {issue.description}\n"
            f"Fix instruction: {issue.suggestion}\n\n"
            f"File: {issue.file}\n"
            f"Context around line {issue.line}:\n{context_snippet}\n\n"
            f"Full file source:\n{source[:_MAX_FILE_CHARS]}"
        )},
    ]
    try:
        raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.05, "num_ctx": 32768})
        raw = raw.strip()
        if raw.startswith("```"):
            lines_raw = raw.splitlines()
            inner, in_block = [], False
            for ln in lines_raw:
                if ln.startswith("```") and not in_block:
                    in_block = True
                    continue
                if ln.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(ln)
            raw = "\n".join(inner)
        data = json.loads(raw)
        return {
            "old_code": data.get("old_code", ""),
            "new_code": data.get("new_code", ""),
            "explanation": data.get("explanation", ""),
        }
    except Exception as e:
        logger.warning(
            "[SelfAnalysis] Patch generation failed for %s:%d: %s",
            issue.file, issue.line, e,
        )
        return {"old_code": "", "new_code": "", "explanation": ""}


def propose_patches(
    issues: list[AnalysisIssue],
    files: list[FileInfo],
    only_blocking: bool = True,
) -> list[AnalysisIssue]:
    """
    For each issue (optionally only blocking), generate a patch.
    Returns issues with .patch field filled.
    """
    file_map = {f.rel_path: f.content for f in files}
    target = [
        i for i in issues
        if (not only_blocking or i.is_blocking) and i.file in file_map
    ]

    for idx, issue in enumerate(target):
        report_progress(
            f"🛠 Генерирую патч {idx+1}/{len(target)}: {issue.file}:{issue.line}..."
        )
        source = file_map[issue.file]
        patch_data = _generate_patch(issue, source)
        if patch_data["old_code"] and patch_data["new_code"]:
            issue.patch = json.dumps(patch_data, ensure_ascii=False)
        else:
            issue.patch = ""

    return issues


# ---------------------------------------------------------------------------
# Step 5: Apply patches
# ---------------------------------------------------------------------------

def apply_patches(
    issues: list[AnalysisIssue],
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Apply patches from issues that have .patch set.
    Returns (applied, skipped) lists of descriptions.
    dry_run=True: validate but don't write files.
    """
    applied: list[str] = []
    skipped: list[str] = []

    for issue in issues:
        if not issue.patch:
            skipped.append(f"{issue.file}:{issue.line} — no safe patch generated")
            continue

        try:
            patch_data = json.loads(issue.patch)
        except json.JSONDecodeError:
            skipped.append(f"{issue.file}:{issue.line} — invalid patch JSON")
            continue

        old_code = patch_data.get("old_code", "")
        new_code = patch_data.get("new_code", "")
        explanation = patch_data.get("explanation", "")

        if not old_code or not new_code:
            skipped.append(f"{issue.file}:{issue.line} — empty patch, skipped")
            continue

        file_path = REPO_ROOT / issue.file
        if not file_path.exists():
            skipped.append(f"{issue.file}:{issue.line} — file not found")
            continue

        current = file_path.read_text(encoding="utf-8")
        if old_code not in current:
            skipped.append(
                f"{issue.file}:{issue.line} — old_code not found in file (source changed?)"
            )
            continue

        # Validate new_code is syntactically valid Python
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            skipped.append(f"{issue.file}:{issue.line} — new_code syntax error: {e}")
            continue

        patched = current.replace(old_code, new_code, 1)

        if not dry_run:
            file_path.write_text(patched, encoding="utf-8")
            logger.info(
                "[SelfAnalysis] Applied patch to %s:%d (%s)",
                issue.file, issue.line, explanation,
            )

            # Record in ProjectContext if available
            try:
                from brain.agents.project_context import ProjectContext
                ctx = ProjectContext.load()
                ctx.add_decision(
                    f"SelfAnalysis patched {issue.file}:{issue.line} — {explanation}"
                )
                ctx.record_test(
                    name=f"self_analysis_{issue.file.replace('/', '_')}_{issue.line}",
                    passed=True,
                    details=explanation,
                )
                ctx.save()
            except Exception as e:
                logger.warning("[SelfAnalysis] ProjectContext update failed: %s", e)

        applied.append(
            f"{issue.file}:{issue.line} — {explanation or issue.description[:60]}"
        )

    return applied, skipped


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

_APPLY_KEYWORDS = re.compile(
    r"(применить|применяй|apply|запиши|исправь|исправить|авто|auto|patch)",
    re.IGNORECASE,
)
_TARGET_DIRS_RE = re.compile(
    r"(brain|tools|core|dev|voice|ui)[/\\]?[\w/]*",
    re.IGNORECASE,
)


def _parse_query(query: str) -> tuple[list[str] | None, bool, bool]:
    """
    Returns (target_dirs, apply_patches, only_blocking).
    """
    apply = bool(_APPLY_KEYWORDS.search(query))
    only_blocking = "medium" not in query.lower() and "low" not in query.lower()
    dirs = _TARGET_DIRS_RE.findall(query)
    target_dirs = [d.lower() for d in dirs] if dirs else None
    return target_dirs, apply, only_blocking


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(query: str, history: list[dict] | None = None) -> str:
    """
    Main entry point for SelfAnalysisAgent.
    Called from brain/ask.py _dispatch when route == "analyze".

    Trigger phrases:
      "проанализируй себя"           -> scan + overview + find issues
      "проанализируй brain/ask.py"   -> scan specific file/dir
      "найди баги и исправь"         -> scan + analyse + apply patches
    """
    target_dirs, should_apply, only_blocking = _parse_query(query)

    report_progress("🔍 Сканирую файлы проекта...")
    files = scan_repo(target_dirs)

    if not files:
        return "❌ Не найдено файлов для анализа."

    syntax_errors = [f for f in files if not f.syntax_ok]
    if syntax_errors:
        report_progress(f"⚠️ Найдено SyntaxError в {len(syntax_errors)} файлах")

    report_progress("🏗 Строю архитектурный обзор...")
    overview = build_overview(files)

    report_progress("🔎 Анализирую исходный код на проблемы...")
    issues = analyse_files(files)

    applied: list[str] = []
    skipped: list[str] = []

    if issues and should_apply:
        report_progress(f"🛠 Генерирую патчи для {len(issues)} проблем...")
        issues = propose_patches(issues, files, only_blocking=only_blocking)
        applied, skipped = apply_patches(issues, dry_run=False)

    # Build response
    lines: list[str] = []

    lines.append("## 🧠 Архитектурный обзор")
    lines.append(overview)
    lines.append("")

    if syntax_errors:
        lines.append("## ❌ SyntaxError (нужна немедленная правка)")
        for f in syntax_errors:
            lines.append(f"  `{f.rel_path}`: {f.syntax_error}")
        lines.append("")

    if issues:
        lines.append(f"## 🔍 Найденные проблемы ({len(issues)} шт.)")
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        for i in sorted(issues, key=lambda x: severity_order.get(x.severity, 99)):
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(i.severity, "⚪")
            patch_status = " ✅ патч применён" if any(i.file in a for a in applied) else ""
            lines.append(f"  {icon} [{i.severity}] `{i.file}:{i.line}` — {i.description}{patch_status}")
            lines.append(f"     Фикс: {i.suggestion}")
        lines.append("")
    else:
        lines.append("## ✅ Проблем не найдено")
        lines.append("")

    if applied:
        lines.append("## ✅ Применённые патчи")
        for a in applied:
            lines.append(f"  - {a}")
        lines.append("")

    if skipped:
        lines.append("## ⚠️ Пропущено (ручная проверка)")
        for s in skipped:
            lines.append(f"  - {s}")

    return "\n".join(lines)
