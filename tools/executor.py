"""tools/executor.py — run Python code/files and pytest in a subprocess."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

JARVIS_ROOT = Path("C:/jarvis")
SANDBOX     = JARVIS_ROOT / "sandbox"
TIMEOUT     = 30   # seconds per run
TEST_TIMEOUT = 60  # seconds for pytest


def run_python(code: str, cwd: str | None = None) -> dict:
    """Execute Python source string in a subprocess. Returns stdout/stderr."""
    work_dir = Path(cwd) if cwd else SANDBOX
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=TIMEOUT, cwd=str(work_dir),
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"Timeout after {TIMEOUT}s"}


def run_file(path: str, args: list[str] | None = None) -> dict:
    """Execute a Python file. path can be absolute or relative to JARVIS_ROOT."""
    p = Path(path)
    if not p.is_absolute():
        p = JARVIS_ROOT / p
    try:
        result = subprocess.run(
            [sys.executable, str(p)] + (args or []),
            capture_output=True, text=True,
            timeout=TIMEOUT, cwd=str(JARVIS_ROOT),
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"Timeout after {TIMEOUT}s"}


def run_pytest(path: str = ".") -> dict:
    """Run pytest and return combined output."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True,
            timeout=TEST_TIMEOUT, cwd=str(JARVIS_ROOT),
        )
        return {
            "ok": result.returncode == 0,
            "passed": result.returncode == 0,
            "output": (result.stdout + result.stderr)[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "passed": False, "output": f"Pytest timeout after {TEST_TIMEOUT}s"}
