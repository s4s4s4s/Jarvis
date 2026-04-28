"""tools/git_ops.py — thin wrappers around git CLI for Jarvis self-modification."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("C:/jarvis")


def _git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(cwd))


def git_status() -> dict:
    r = _git("status", "--short")
    return {"ok": r.returncode == 0, "output": r.stdout}


def git_diff(path: str | None = None) -> dict:
    args = ["diff"] + ([path] if path else [])
    r = _git(*args)
    return {"ok": True, "diff": r.stdout[-5000:]}


def git_commit(message: str, add_all: bool = True) -> dict:
    if add_all:
        _git("add", "-A")
    r = _git("commit", "-m", message)
    return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}


def git_push() -> dict:
    r = _git("push")
    return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}


def git_stash(message: str = "") -> dict:
    args = ["stash", "push"] + (["-m", message] if message else [])
    r = _git(*args)
    return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}
