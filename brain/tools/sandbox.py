from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

# Python stdlib modules — never need pip install
_STDLIB_MODULES = {
    "os", "sys", "re", "json", "time", "datetime", "math", "random",
    "hashlib", "pathlib", "shutil", "subprocess", "tempfile", "threading",
    "asyncio", "logging", "argparse", "collections", "itertools", "functools",
    "typing", "dataclasses", "enum", "abc", "copy", "io", "struct",
    "sqlite3", "csv", "configparser", "pickle", "base64", "uuid",
    "socket", "ssl", "http", "urllib", "email", "html", "xml",
    "unittest", "contextlib", "weakref", "gc", "traceback", "inspect",
    "ast", "dis", "importlib", "pkgutil", "platform", "signal",
    "queue", "heapq", "bisect", "array", "statistics", "decimal", "fractions",
    "string", "textwrap", "difflib", "pprint", "reprlib",
    "glob", "fnmatch", "fileinput", "zipfile", "tarfile", "gzip", "bz2",
    "hmac", "secrets", "getpass", "curses", "tkinter",
    "multiprocessing", "concurrent", "selectors", "select",
    "ctypes", "mmap", "msvcrt", "winreg",
}

SANDBOX_TIMEOUT = 8  # seconds

# Keywords that indicate a script has infinite loops — we allow timeout
_INFINITE_LOOP_HINTS = (
    "while True",
    "while 1",
    "asyncio.run",
    "app.run",
    "bot.polling",
    "updater.start_polling",
    "application.run_polling",
)


def has_infinite_loop(code: str) -> bool:
    """Heuristic: detect scripts that are expected to run forever."""
    return any(hint in code for hint in _INFINITE_LOOP_HINTS)


def extract_pip_requirements(code: str) -> list[str]:
    """
    Parse import statements and return only non-stdlib packages
    that likely need pip install.
    Maps common import names to pip package names.
    """
    import ast as _ast

    _IMPORT_TO_PIP = {
        "telegram": "python-telegram-bot",
        "telebot": "pyTelegramBotAPI",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "bs4": "beautifulsoup4",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
        "psutil": "psutil",
        "requests": "requests",
        "aiohttp": "aiohttp",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "sqlalchemy": "SQLAlchemy",
        "alembic": "alembic",
        "celery": "celery",
        "redis": "redis",
        "pymongo": "pymongo",
        "flask": "flask",
        "django": "django",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "torch": "torch",
        "transformers": "transformers",
        "openai": "openai",
        "anthropic": "anthropic",
        "httpx": "httpx",
        "paramiko": "paramiko",
        "cryptography": "cryptography",
        "jwt": "PyJWT",
        "passlib": "passlib",
        "bcrypt": "bcrypt",
    }

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []

    packages: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _STDLIB_MODULES:
                    packages.add(_IMPORT_TO_PIP.get(root, root))
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in _STDLIB_MODULES:
                    packages.add(_IMPORT_TO_PIP.get(root, root))

    return sorted(packages)


def run_in_sandbox(
    code: str,
    timeout: int = SANDBOX_TIMEOUT,
    auto_install: bool = True,
) -> tuple[str, str, int]:
    """
    Run Python code in a subprocess sandbox.

    Returns:
        (stdout, stderr, returncode)
        returncode == -1  means timeout (expected for infinite loops)
        returncode == -2  means pip install failed
    """
    # Auto-install missing packages before running
    if auto_install:
        packages = extract_pip_requirements(code)
        for pkg in packages:
            try:
                logger.debug("[Sandbox] Installing %s...", pkg)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    timeout=30,
                    capture_output=True,
                )
            except Exception as e:
                logger.warning("[Sandbox] pip install %s failed: %s", pkg, e)

    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        logger.debug("[Sandbox] Timeout after %ds (expected for long-running scripts)", timeout)
        return "", f"[Sandbox] Timeout after {timeout}s", -1
    except Exception as e:
        return "", str(e), -2
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def sandbox_audit(code: str, timeout: int = SANDBOX_TIMEOUT) -> tuple[str, bool]:
    """
    Run code in sandbox and return (report, has_issues).

    Logic:
    - Timeout with no stderr  → OK (infinite loop, expected)
    - Timeout with stderr      → issues (crashed before timeout)
    - returncode == 0          → OK
    - returncode != 0          → issues, report contains stderr/traceback
    """
    is_infinite = has_infinite_loop(code)
    stdout, stderr, returncode = run_in_sandbox(code, timeout=timeout)

    if returncode == 0:
        report = f"[Sandbox] ✅ Ran successfully." + (f" Output:\n{stdout[:500]}" if stdout else "")
        return report, False

    if returncode == -1:  # Timeout
        if is_infinite and not stderr.strip().replace(f"[Sandbox] Timeout after {timeout}s", "").strip():
            # Expected infinite loop, no real errors
            report = f"[Sandbox] ✅ Timeout after {timeout}s — expected (script has infinite loop/polling)."
            return report, False
        else:
            # Timed out but there's stderr — something crashed
            report = f"[Sandbox] ⚠️ Timeout with errors:\n{stderr[:1000]}"
            return report, bool(stderr.strip())

    # Non-zero exit
    error_text = stderr.strip() or stdout.strip()
    report = f"[Sandbox] ❌ Runtime error (exit {returncode}):\n{error_text[:1500]}"
    return report, True
