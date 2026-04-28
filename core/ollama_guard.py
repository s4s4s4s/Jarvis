"""core/ollama_guard.py

Auto-start Ollama before Jarvis launches.
If Ollama is already running - does nothing.
If not - launches `ollama serve` as background process and waits.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_URL    = "http://localhost:11434"
_HEALTH_URL    = _OLLAMA_URL + "/api/tags"
_STARTUP_WAIT  = 30
_POLL_INTERVAL = 0.5
_HTTP_TIMEOUT  = 3


def _is_running() -> bool:
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.get(_HEALTH_URL)
            return r.status_code == 200
    except Exception:
        return False


def _find_ollama_exe() -> str | None:
    import shutil
    found = shutil.which("ollama")
    if found:
        return found

    username = os.getenv("USERNAME", "")
    candidate1 = os.path.join("C:\\Users", username, "AppData", "Local", "Programs", "Ollama", "ollama.exe")
    candidate2 = "C:\\Program Files\\Ollama\\ollama.exe"
    candidate3 = "C:\\ollama\\ollama.exe"

    for path in [candidate1, candidate2, candidate3]:
        if os.path.isfile(path):
            return path

    return None


def ensure_ollama() -> bool:
    """
    Make sure Ollama is running. Launch it if not.

    Returns:
        True  - Ollama is ready
        False - could not start (no exe, timeout, etc.)
    """
    if _is_running():
        logger.info("[ollama_guard] Ollama already running")
        return True

    logger.info("[ollama_guard] Ollama not running - trying to start...")
    print("[Jarvis] Ollama не запущена - запускаю автоматически...", flush=True)

    exe = _find_ollama_exe()
    if exe is None:
        logger.error("[ollama_guard] ollama.exe not found")
        print(
            "[Jarvis] ollama.exe не найден.\n"
            "        Скачай и установи: https://ollama.com/download\n"
            "        Затем перезапусти Jarvis.",
            flush=True,
        )
        return False

    logger.info("[ollama_guard] Launching: %s serve", exe)
    try:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
    except Exception as e:
        logger.error("[ollama_guard] Failed to launch Ollama: %s", e)
        print("[Jarvis] Не удалось запустить Ollama: " + str(e), flush=True)
        return False

    deadline = time.monotonic() + _STARTUP_WAIT
    dots = 0
    while time.monotonic() < deadline:
        if _is_running():
            elapsed = round(_STARTUP_WAIT - (deadline - time.monotonic()), 1)
            logger.info("[ollama_guard] Ollama ready in %ss", elapsed)
            print("\n[Jarvis] Ollama запущена!", flush=True)
            return True
        dots += 1
        print("\r[Jarvis] Жду Ollama" + "." * (dots % 4) + "   ", end="", flush=True)
        time.sleep(_POLL_INTERVAL)

    logger.error("[ollama_guard] Ollama did not respond in %ds", _STARTUP_WAIT)
    print(
        "\n[Jarvis] Ollama не ответила за " + str(_STARTUP_WAIT) + "с. "
        "Запусти `ollama serve` вручную и перезапусти Jarvis.",
        flush=True,
    )
    return False
