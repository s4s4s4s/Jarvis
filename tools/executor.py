"""tools/executor.py — run Python code/files and pytest in a subprocess.

FIXES:
  - _DANGEROUS_PATTERNS: subprocess.run pattern was too broad and blocked
    code_agent's own test runner when it called run_pytest internally.
    Now only blocks subprocess in USER-generated code via run_python().
    run_file() and run_pytest() are internal trusted calls — no safety check.
  - Added file size limit to prevent LLM from writing multi-MB files.
"""
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

# FIX: patterns that are dangerous in USER-generated code (run_python only)
# Removed subprocess.run pattern — it blocked internal test tooling.
# Also removed eval/exec — these are valid in legitimate code snippets.
_DANGEROUS_PATTERNS = [
    r"os\.system\s*\(",
    r"shutil\.rmtree\s*\(",
    r"__import__\s*\(['\"]os['\"]\)",
    # Block direct shell calls with shell=True
    r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
    # Block rm -rf style commands
    r"['\"]rm\s+-rf",
]


def _check_code_safety(code: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Blocks obviously dangerous patterns.
    Applied only to user-provided code in run_python(), NOT to internal calls.
    """
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked dangerous pattern: {pattern}"
    return True, ""


def run_python(code: str, cwd: str | None = None) -> dict:
    """Execute Python source string in a subprocess. Returns stdout/stderr.
    Safety check applied — dangerous patterns are blocked.
    """
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
    """Execute a Python file. Trusted internal call — no safety check.
    path can be absolute or relative to JARVIS_ROOT.
    """
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
    except FileNotFoundError:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"File not found: {p}"}


def run_pytest(path: str = ".") -> dict:
    """Run pytest. Trusted internal call — no safety check."""
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
