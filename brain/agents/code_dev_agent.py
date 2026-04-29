"""brain/agents/code_dev_agent.py

CodeDevAgent — Universal code developer.

Works with ANY codebase — not just Jarvis.
Jarvis uses this on himself as a learning exercise, but the same agent
can be pointed at any local folder or GitHub repo.

Capabilities:
  1. read_project(path|url)  — load source files from local dir or GitHub URL
  2. analyse(files)          — find bugs, missing features, integration gaps
  3. fix_bug(issue, files)   — generate + apply minimal patch
  4. add_feature(spec, files)— scaffold new feature into existing codebase
  5. run_tests(path)         — execute test suite, return results
  6. full_cycle(request)     — plan → analyse → implement → test → commit

Entry point: run(query, history) — called from _dispatch when route=="develop"

Trigger phrases:
  "разработай ..." / "develop ..."    → full_cycle
  "проанализируй <path>"            → analyse only
  "добавь фичу ..."                   → add_feature
  "исправь баг / почини ..."        → fix_bug
  "запусти тесты ..."               → run_tests
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.client import chat, MODEL_HEAVY, MODEL_FAST
from brain.ask import report_progress

logger = logging.getLogger(__name__)

_MAX_FILE_CHARS = 8000
_MAX_FILES_PER_BATCH = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProjectFile:
    path: str          # relative path from project root
    content: str
    language: str = "python"
    syntax_ok: bool = True
    syntax_error: str = ""


@dataclass
class DevIssue:
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW
    file: str
    line: int
    description: str
    fix: str
    patch_old: str = ""
    patch_new: str = ""


@dataclass
class DevResult:
    project_root: str
    files_read: int = 0
    issues_found: list[DevIssue] = field(default_factory=list)
    features_added: list[str] = field(default_factory=list)
    patches_applied: list[str] = field(default_factory=list)
    test_output: str = ""
    branch_url: str = ""

    def to_markdown(self) -> str:
        lines = [f"## 💻 Проект: `{self.project_root}`"]
        lines.append(f"Файлов прочитано: {self.files_read}")

        if self.issues_found:
            lines.append(f"\n### 🔍 Найдено проблем: {len(self.issues_found)}")
            sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            for i in sorted(self.issues_found, key=lambda x: severity_order.get(x.severity, 9)):
                icon = sev_icon.get(i.severity, "⚪")
                lines.append(f"  {icon} `{i.file}:{i.line}` — {i.description}")
                lines.append(f"     Фикс: {i.fix}")

        if self.patches_applied:
            lines.append(f"\n### ✅ Применено патчей: {len(self.patches_applied)}")
            for p in self.patches_applied:
                lines.append(f"  - {p}")

        if self.features_added:
            lines.append(f"\n### ✨ Добавлено фич: {len(self.features_added)}")
            for f in self.features_added:
                lines.append(f"  - {f}")

        if self.test_output:
            lines.append(f"\n### 🧪 Тесты")
            lines.append(f"```\n{self.test_output[:2000]}\n```")

        if self.branch_url:
            lines.append(f"\n### 🚀 Пуш\n  [{self.branch_url}]({self.branch_url})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: Read project files
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".cs": "csharp",
    ".cpp": "cpp", ".c": "c", ".rb": "ruby", ".php": "php",
    ".sh": "bash", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".md": "markdown",
}
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
              ".mypy_cache", "dist", "build", ".next", ".nuxt"}
_MAX_TOTAL_FILES = 50


def read_local_project(root: str | Path, extensions: list[str] | None = None) -> list[ProjectFile]:
    """Read all code files from a local directory."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    allowed = set(extensions) if extensions else set(_CODE_EXTENSIONS.keys())
    files: list[ProjectFile] = []

    for p in sorted(root.rglob("*")):
        if len(files) >= _MAX_TOTAL_FILES:
            break
        if p.is_dir() or any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix not in allowed:
            continue
        try:
            rel = p.relative_to(root).as_posix()
            content = p.read_text(encoding="utf-8", errors="replace")
            lang = _CODE_EXTENSIONS.get(p.suffix, "text")
            syntax_ok, syntax_err = True, ""
            if p.suffix == ".py":
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_ok, syntax_err = False, str(e)
            files.append(ProjectFile(
                path=rel, content=content, language=lang,
                syntax_ok=syntax_ok, syntax_error=syntax_err,
            ))
        except Exception as e:
            logger.warning("[CodeDev] Failed to read %s: %s", p, e)

    logger.info("[CodeDev] Read %d files from %s", len(files), root)
    return files


def read_github_project(repo: str, subdir: str = "", branch: str = "main") -> list[ProjectFile]:
    """
    Read files from a GitHub repo.
    repo format: "owner/repo"
    Uses github_reader (API or git clone fallback).
    """
    # BUG-FIX: import is no longer conditional — github_reader always exists now
    from brain.agents.github_reader import read_repo_files
    return read_repo_files(repo, subdir=subdir, branch=branch)


# ---------------------------------------------------------------------------
# Step 2: Analyse files for issues
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM = """\
You are a senior software engineer doing a thorough code review.

You receive source files from a project. Find every real issue:
  1. Bugs causing runtime errors or wrong behaviour
  2. Logic errors — wrong conditions, missing edge cases, broken flows
  3. Security vulnerabilities — injection, hardcoded secrets, unsafe operations
  4. Performance problems — N+1 queries, unbounded loops, memory leaks
  5. Missing error handling — silent failures, uncaught exceptions
  6. Integration gaps — components that should connect but don't
  7. Dead code, unreachable branches, unused imports
  8. Missing features referenced in comments or TODOs

For each issue output ONE line:
  [SEVERITY] filename.ext:LINE — description — fix: concrete fix instruction

SEVERITY: CRITICAL / HIGH / MEDIUM / LOW
LINE: exact line number, 0 if file-level

If no issues: output LGTM
No prose, no markdown. Only issue lines or LGTM.
"""


def analyse_project(files: list[ProjectFile]) -> list[DevIssue]:
    """Send files in batches to LLM, collect all issues."""
    all_issues: list[DevIssue] = []
    batches = [files[i:i+_MAX_FILES_PER_BATCH] for i in range(0, len(files), _MAX_FILES_PER_BATCH)]

    for batch_idx, batch in enumerate(batches):
        report_progress(f"🔍 Анализирую батч {batch_idx+1}/{len(batches)}...")
        parts = []
        for f in batch:
            content = f.content[:_MAX_FILE_CHARS]
            if len(f.content) > _MAX_FILE_CHARS:
                content += f"\n... [TRUNCATED]"
            parts.append(f"=== {f.path} ===\n{content}")

        messages = [
            {"role": "system", "content": _ANALYSIS_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        try:
            raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.05, "num_ctx": 32768})
            issues = _parse_issues(raw, {f.path for f in batch})
            all_issues.extend(issues)
        except Exception as e:
            logger.error("[CodeDev] Analysis batch %d failed: %s", batch_idx+1, e)

    return all_issues


def _parse_issues(raw: str, known_files: set[str]) -> list[DevIssue]:
    issues: list[DevIssue] = []
    short_to_full = {Path(f).name: f for f in known_files}

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "LGTM":
            continue
        m = re.match(
            r"\[(CRITICAL|HIGH|MEDIUM|LOW)\]\s+"
            r"([\w./\-]+\.[\w]+)(?::(\d+))?\s*[\u2014\-]+\s*"
            r"(.+?)(?:\s*[\u2014\-]+\s*fix:\s*(.+))?",
            line, re.IGNORECASE,
        )
        if m:
            raw_file = m.group(2)
            resolved = raw_file
            if raw_file not in known_files:
                resolved = short_to_full.get(Path(raw_file).name, raw_file)
            issues.append(DevIssue(
                severity=m.group(1).upper(),
                file=resolved,
                line=int(m.group(3)) if m.group(3) else 0,
                description=m.group(4).strip(),
                fix=m.group(5).strip() if m.group(5) else "see description",
            ))
    return issues


# ---------------------------------------------------------------------------
# Step 3: Fix bugs
# ---------------------------------------------------------------------------

_FIXER_SYSTEM = """\
You are a senior engineer generating a minimal, safe code patch.

Output JSON only:
{
  "old_code": "exact substring to replace (must exist verbatim in the file)",
  "new_code": "replacement with the fix",
  "explanation": "one sentence"
}

Rules:
  - old_code MUST be an exact copy-paste from the file — no paraphrasing
  - new_code must be syntactically valid
  - Change only what is needed
  - If unsafe to auto-patch, return empty strings for old_code and new_code
  - No markdown, no prose — only JSON
"""


def generate_patch(issue: DevIssue, source: str) -> tuple[str, str, str]:
    """Returns (old_code, new_code, explanation). Empty strings if unsafe."""
    lines = source.splitlines()
    start = max(0, issue.line - 10)
    end = min(len(lines), issue.line + 20)
    ctx = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[start:end], start=start))

    messages = [
        {"role": "system", "content": _FIXER_SYSTEM},
        {"role": "user", "content": (
            f"Issue: [{issue.severity}] {issue.description}\n"
            f"Fix: {issue.fix}\n\n"
            f"File: {issue.file} (lines {start+1}-{end}):\n{ctx}\n\n"
            f"Full source:\n{source[:_MAX_FILE_CHARS]}"
        )},
    ]
    try:
        raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.05, "num_ctx": 32768})
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[\w]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
        data = json.loads(raw)
        return (
            data.get("old_code", ""),
            data.get("new_code", ""),
            data.get("explanation", ""),
        )
    except Exception as e:
        logger.warning("[CodeDev] Patch generation failed: %s", e)
        return "", "", ""


def apply_patch_to_file(
    file_path: Path,
    old_code: str,
    new_code: str,
    dry_run: bool = False,
) -> bool:
    """Apply old_code -> new_code patch. Returns True if applied."""
    if not file_path.exists():
        return False
    current = file_path.read_text(encoding="utf-8")
    if old_code not in current:
        logger.warning("[CodeDev] old_code not found in %s", file_path)
        return False
    # Validate syntax for Python files
    if file_path.suffix == ".py":
        patched = current.replace(old_code, new_code, 1)
        try:
            ast.parse(patched)
        except SyntaxError as e:
            logger.warning("[CodeDev] patched file syntax error in %s: %s", file_path, e)
            return False
    if not dry_run:
        patched = current.replace(old_code, new_code, 1)
        file_path.write_text(patched, encoding="utf-8")
    return True


def fix_issues(
    issues: list[DevIssue],
    project_root: Path,
    file_map: dict[str, str],
    only_blocking: bool = True,
) -> list[str]:
    """Fix all (blocking) issues. Returns list of applied descriptions."""
    applied: list[str] = []
    targets = [i for i in issues if (not only_blocking or i.severity in ("CRITICAL", "HIGH"))]

    for idx, issue in enumerate(targets):
        report_progress(f"🛠 Фикшу {idx+1}/{len(targets)}: {issue.file}:{issue.line}")
        source = file_map.get(issue.file, "")
        if not source:
            continue
        old, new, explanation = generate_patch(issue, source)
        if not old or not new:
            continue
        file_path = project_root / issue.file
        if apply_patch_to_file(file_path, old, new):
            applied.append(f"{issue.file}:{issue.line} — {explanation}")
            file_map[issue.file] = file_map[issue.file].replace(old, new, 1)

    return applied


# ---------------------------------------------------------------------------
# Step 4: Add new feature
# ---------------------------------------------------------------------------

_FEATURE_PLANNER_SYSTEM = """\
You are a senior engineer implementing a new feature into an existing codebase.

You receive:
  1. Feature specification (what to build)
  2. Existing project structure (file list + key functions)

Output a JSON plan:
{
  "summary": "one sentence what will be built",
  "new_files": [
    {"path": "relative/path.py", "description": "what this file does", "content": "full file content"}
  ],
  "modified_files": [
    {"path": "existing/file.py", "old_code": "exact substring", "new_code": "replacement", "reason": "why"}
  ]
}

Rules:
  - new_files must have complete, working content
  - modified_files old_code must be exact substrings of existing files
  - Follow the coding style of the existing codebase
  - Keep changes minimal and focused
  - No markdown in JSON strings
  - Output only JSON
"""


def add_feature(
    spec: str,
    project_root: Path,
    files: list[ProjectFile],
) -> list[str]:
    """Scaffold a new feature. Returns list of created/modified file descriptions."""
    structure = "\n".join(
        f"{f.path} ({f.language}, {len(f.content)} chars)"
        for f in files[:30]
    )
    file_map = {f.path: f.content for f in files}

    messages = [
        {"role": "system", "content": _FEATURE_PLANNER_SYSTEM},
        {"role": "user", "content": (
            f"Feature spec: {spec}\n\n"
            f"Project structure:\n{structure}\n\n"
            f"Key files (first 2000 chars each):\n"
            + "\n\n".join(
                f"=== {f.path} ===\n{f.content[:2000]}"
                for f in files[:8]
            )
        )},
    ]

    try:
        raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.1, "num_ctx": 32768})
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[\w]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
        plan = json.loads(raw)
    except Exception as e:
        logger.error("[CodeDev] Feature plan parse failed: %s", e)
        return [f"❌ Не удалось сгенерировать план: {e}"]

    results: list[str] = []

    for nf in plan.get("new_files", []):
        try:
            fpath = project_root / nf["path"]
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(nf["content"], encoding="utf-8")
            results.append(f"✨ Создан: {nf['path']} — {nf.get('description', '')}")
            logger.info("[CodeDev] Created %s", fpath)
        except Exception as e:
            results.append(f"❌ Ошибка при создании {nf.get('path', '?')}: {e}")

    for mf in plan.get("modified_files", []):
        fpath = project_root / mf["path"]
        old = mf.get("old_code", "")
        new = mf.get("new_code", "")
        if old and new and apply_patch_to_file(fpath, old, new):
            results.append(f"✏️ Изменён: {mf['path']} — {mf.get('reason', '')}")
        else:
            results.append(f"⚠️ Не удалось патчить: {mf.get('path', '?')}")

    return results


# ---------------------------------------------------------------------------
# Step 5: Run tests
# ---------------------------------------------------------------------------

def run_tests(project_root: Path, timeout: int = 60) -> str:
    """Run pytest (or other test runner) and return output."""
    runners = [
        ["python", "-m", "pytest", "-x", "--tb=short", "-q"],
        ["python", "-m", "unittest", "discover"],
        ["npm", "test", "--", "--watchAll=false"],
        ["cargo", "test"],
        ["go", "test", "./..."],
    ]
    runner_indicators = {
        "pytest.ini": runners[0], "setup.cfg": runners[0],
        "package.json": runners[2], "Cargo.toml": runners[3],
        "go.mod": runners[4],
    }

    cmd = runners[0]
    for indicator, r in runner_indicators.items():
        if (project_root / indicator).exists():
            cmd = r
            break

    try:
        result = subprocess.run(
            cmd, cwd=str(project_root),
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        return output[:3000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Tests timed out after {timeout}s"
    except FileNotFoundError:
        return f"Test runner not found: {cmd[0]}"
    except Exception as e:
        return f"Test runner error: {e}"


# ---------------------------------------------------------------------------
# Step 6: Git commit
# ---------------------------------------------------------------------------

def git_commit(project_root: Path, message: str, push: bool = False) -> str:
    """Stage all changes and commit. Optionally push."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(project_root), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], cwd=str(project_root), check=True, capture_output=True)
        if push:
            r = subprocess.run(["git", "push"], cwd=str(project_root), capture_output=True, text=True)
            if r.returncode != 0:
                return f"⚠️ commit OK, push failed: {r.stderr[:200]}"
        return f"✅ commit: {message}"
    except subprocess.CalledProcessError as e:
        return f"❌ git error: {e.stderr.decode()[:200] if e.stderr else str(e)}"


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

_MODE_PATTERNS = {
    "fix":     re.compile(r"(исправь|фиксани|почини|fix|patch|bug)", re.I),
    "feature": re.compile(r"(добавь|допиши|реализуй|сделай|feature|add|implement|scaffold)", re.I),
    "test":    re.compile(r"(запусти тест|test|pytest|unittest)", re.I),
    "analyse": re.compile(r"(анализируй|проверь|analyse|analyze|review)", re.I),
}
_PATH_PATTERN = re.compile(r"[A-Za-z]:[/\\][\w/\\. -]+|/[\w/. -]{3,}|\.[/\\][\w/\\. -]+")
# BUG-FIX: also detect GitHub repo URLs like github.com/owner/repo or owner/repo pattern
_GITHUB_REPO_RE = re.compile(
    r"github\.com/([\w-]+/[\w.-]+)|\b([\w-]+/[\w.-]+)\b(?=.*github|\.git|\.py|\.js|\.ts)",
    re.IGNORECASE,
)


def _parse_query(query: str) -> tuple[str, str, str]:
    """
    Returns (mode, project_path, extra_spec).
    mode: full_cycle | fix | feature | test | analyse
    project_path: local path OR "owner/repo" for GitHub
    """
    mode = "full_cycle"
    for m, pattern in _MODE_PATTERNS.items():
        if pattern.search(query):
            mode = m
            break

    # Check for GitHub repo first
    gh_match = _GITHUB_REPO_RE.search(query)
    if gh_match:
        project_path = (gh_match.group(1) or gh_match.group(2)).strip()
        spec = re.sub(_GITHUB_REPO_RE.pattern, "", query, flags=re.IGNORECASE).strip()
        return mode, f"github:{project_path}", spec

    # Local path
    path_match = _PATH_PATTERN.search(query)
    project_path = path_match.group(0).strip() if path_match else ""

    spec = re.sub(
        r"(добавь'?|feature|add|implement|scaffold|допиши|реализуй)\s+",
        "", query, flags=re.I,
    ).strip()

    return mode, project_path, spec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(query: str, history: list[dict] | None = None) -> str:
    """
    Universal code developer entry point.
    Called from _dispatch when route == "develop".
    """
    mode, project_path, spec = _parse_query(query)

    # BUG-FIX: handle GitHub repos detected in query
    if project_path.startswith("github:"):
        repo_id = project_path[len("github:"):]
        report_progress(f"📂 Читаю GitHub репо: {repo_id}...")
        try:
            files = read_github_project(repo_id)
        except Exception as e:
            return f"❌ Не удалось прочитать GitHub репо {repo_id}: {e}"
        root = Path(f"github:{repo_id}")
    elif project_path:
        root = Path(project_path).expanduser()
        report_progress(f"📂 Читаю файлы: {root}...")
        try:
            files = read_local_project(root)
        except FileNotFoundError as e:
            return f"❌ {e}"
    else:
        # Default: Jarvis own repo root
        root = Path(__file__).resolve().parent.parent.parent
        report_progress(f"📂 Читаю собственный код...")
        try:
            files = read_local_project(root)
        except FileNotFoundError as e:
            return f"❌ {e}"

    result = DevResult(project_root=str(root))
    result.files_read = len(files)
    file_map = {f.path: f.content for f in files}

    if mode in ("analyse", "full_cycle", "fix"):
        report_progress(f"🔍 Анализирую {len(files)} файлов...")
        result.issues_found = analyse_project(files)

    if mode in ("fix", "full_cycle") and result.issues_found:
        report_progress(f"🛠 Фикшу {len(result.issues_found)} проблем...")
        if not project_path.startswith("github:"):
            result.patches_applied = fix_issues(
                result.issues_found, root, file_map, only_blocking=True,
            )
        else:
            result.patches_applied = [f"[GitHub mode] фиксы не применяются автоматически (нет write access)"]

    if mode in ("feature", "full_cycle") and spec:
        report_progress(f"✨ Добавляю фичу: {spec[:60]}...")
        if not project_path.startswith("github:"):
            result.features_added = add_feature(spec, root, files)
        else:
            result.features_added = ["[GitHub mode] scaffold не применяется без write access"]

    if mode in ("test", "full_cycle"):
        if not project_path.startswith("github:"):
            report_progress("🧪 Запускаю тесты...")
            result.test_output = run_tests(root)
        else:
            result.test_output = "(тесты не запускаются для удалённых репо)"

    # Auto-commit if anything changed (local only)
    if result.patches_applied or result.features_added:
        if not project_path.startswith("github:"):
            n_fixes = len(result.patches_applied)
            n_feat = len(result.features_added)
            msg = f"chore: CodeDevAgent — {n_fixes} fix(es), {n_feat} feature(s)"
            report_progress("🚀 Коммичу...")
            git_result = git_commit(root, msg)
            result.branch_url = git_result

    return result.to_markdown()
