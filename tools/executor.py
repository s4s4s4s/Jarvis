"""tools/executor.py — run Python code/files and pytest in a subprocess."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# FIX BUG-12: read JARVIS_ROOT from env so it works on Linux/Mac too
JARVIS_ROOT  = Path(os.getenv("JARVIS_ROOT", "C:/jarvis"))
SANDBOX      = JARVIS_ROOT / "sandbox"
TIMEOUT      = 30
TEST_TIMEOUT = 60

# FIX BUG-13: basic deny-list for dangerous patterns in generated code
_DANGEROUS_PATTERNS = [
    r"os\.system\s*\(",
    r"subprocess\.(?:call|run|Popen)\s*\(",
    r"shutil\.rmtree\s*\(",
    r"__import__\s*\(['\"]os['\"]\)",
    r"eval\s*\(",
    r"exec\s*\(",
]


def _check_code_safety(code: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Blocks obviously dangerous patterns."""
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked dangerous pattern: {pattern}"
    return True, ""


def run_python(code: str, cwd: str | None = None) -> dict:
    """Execute Python source string in a subprocess. Returns stdout/stderr."""
    # FIX BUG-13: safety check before execution
    safe, reason = _check_code_safety(code)
    if not safe:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"[Security] {reason}"}

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
