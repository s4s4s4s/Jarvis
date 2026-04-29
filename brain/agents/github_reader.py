"""brain/agents/github_reader.py

Read files from a GitHub repository using the GitHub API.
Used by CodeDevAgent when repo is specified as owner/repo.

Fallback path: git clone via subprocess (used when GITHUB_TOKEN not set).
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.agents.code_dev_agent import ProjectFile

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".cpp": "cpp", ".c": "c", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".md": "markdown",
}
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}
_MAX_FILES = 50
_MAX_FILE_BYTES = 100_000  # 100 KB per file


def read_repo_files(
    repo: str,
    subdir: str = "",
    branch: str = "main",
) -> list["ProjectFile"]:
    """
    Read source files from a GitHub repo.
    repo: "owner/repo" format.
    Returns list of ProjectFile.
    Tries PyGithub first, falls back to git clone.
    """
    from brain.agents.code_dev_agent import ProjectFile

    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        try:
            return _read_via_api(repo, subdir, branch, token)
        except Exception as e:
            logger.warning("[GithubReader] API read failed (%s), falling back to clone", e)

    return _read_via_clone(repo, subdir, branch)


def _read_via_api(
    repo: str,
    subdir: str,
    branch: str,
    token: str,
) -> list["ProjectFile"]:
    """Read using GitHub REST API — no git clone needed."""
    import urllib.request
    import json
    from brain.agents.code_dev_agent import ProjectFile

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Jarvis-CodeDevAgent/1.0",
    }

    def api_get(url: str) -> dict | list:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    base_path = subdir.strip("/") if subdir else ""
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    tree_data = api_get(url)
    items = tree_data.get("tree", [])

    files: list["ProjectFile"] = []
    import ast as _ast

    for item in items:
        if item.get("type") != "blob":
            continue
        path = item["path"]
        if base_path and not path.startswith(base_path):
            continue
        if any(skip in path.split("/") for skip in _SKIP_DIRS):
            continue
        suffix = Path(path).suffix
        if suffix not in _CODE_EXTENSIONS:
            continue
        if item.get("size", 0) > _MAX_FILE_BYTES:
            continue
        if len(files) >= _MAX_FILES:
            break

        try:
            blob = api_get(f"https://api.github.com/repos/{repo}/git/blobs/{item['sha']}")
            content = base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
            lang = _CODE_EXTENSIONS[suffix]
            syntax_ok, syntax_err = True, ""
            if suffix == ".py":
                try:
                    _ast.parse(content)
                except SyntaxError as e:
                    syntax_ok, syntax_err = False, str(e)
            files.append(ProjectFile(
                path=path,
                content=content,
                language=lang,
                syntax_ok=syntax_ok,
                syntax_error=syntax_err,
            ))
        except Exception as e:
            logger.warning("[GithubReader] Failed to fetch blob %s: %s", path, e)

    logger.info("[GithubReader] Read %d files via API from %s", len(files), repo)
    return files


def _read_via_clone(
    repo: str,
    subdir: str,
    branch: str,
) -> list["ProjectFile"]:
    """Fallback: git clone to temp dir and read locally."""
    from brain.agents.code_dev_agent import read_local_project

    with tempfile.TemporaryDirectory() as tmp:
        url = f"https://github.com/{repo}.git"
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--branch", branch, url, tmp],
            capture_output=True, timeout=90,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:300]
            raise RuntimeError(f"git clone failed: {stderr}")
        root = Path(tmp) / subdir if subdir else Path(tmp)
        files = read_local_project(root)
        # Normalize paths to be relative to repo root
        for f in files:
            try:
                full = root / f.path
                f.path = full.relative_to(Path(tmp)).as_posix()
            except Exception:
                pass
        logger.info("[GithubReader] Read %d files via clone from %s", len(files), repo)
        return files
