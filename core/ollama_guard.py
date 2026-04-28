"""core/ollama_guard.py

Автозапуск Ollama перед стартом Jarvis.
Если Ollama уже запущена — ничего не делает.
Если нет — запускает `ollama serve` как фоновый процесс и ждёт готовности.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_URL     = "http://localhost:11434"
_HEALTH_URL     = f"{_OLLAMA_URL}/api/tags"
_STARTUP_WAIT   = 30   # максимум секунд ждать после запуска
_POLL_INTERVAL  = 0.5  # как часто проверять
_HTTP_TIMEOUT   = 3


def _is_running() -> bool:
    """True если Ollama отвечает на /api/tags."""
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.get(_HEALTH_URL)
            return r.status_code == 200
    except Exception:
        return False


def _find_ollama_exe() -> str | None:
    """Ищет ollama.exe: PATH, стандартные пути установки."""
    import shutil
    found = shutil.which("ollama")
    if found:
        return found

    candidates = [
        r"C:\Users\" + os.getenv("USERNAME", "") + r"\AppData\Local\Programs\Ollama\ollama.exe",
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\ollama\ollama.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def ensure_ollama() -> bool:
    """
    Убедиться что Ollama запущена. Если нет — запустить.

    Returns:
        True  — Ollama готова (уже была или успешно запущена)
        False — не удалось запустить (нет exe, таймаут, etc.)
    """
    if _is_running():
        logger.info("[ollama_guard] Ollama already running ✓")
        return True

    logger.info("[ollama_guard] Ollama not running — trying to start...")
    print("[Jarvis] Ollama не запущена — запускаю автоматически...", flush=True)

    exe = _find_ollama_exe()
    if exe is None:
        logger.error("[ollama_guard] ollama.exe not found. Install from https://ollama.com")
        print(
            "[Jarvis] ❌ ollama.exe не найден.\n"
            "        Скачай и установи: https://ollama.com/download\n"
            "        Затем перезапусти Jarvis.",
            flush=True,
        )
        return False

    logger.info("[ollama_guard] Launching: %s serve", exe)
    try:
        # DETACHED_PROCESS + CREATE_NO_WINDOW — окно не появляется, процесс живёт после Jarvis
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
        print(f"[Jarvis] ❌ Не удалось запустить Ollama: {e}", flush=True)
        return False

    # Ждём готовности
    deadline = time.monotonic() + _STARTUP_WAIT
    dots = 0
    while time.monotonic() < deadline:
        if _is_running():
            elapsed = _STARTUP_WAIT - (deadline - time.monotonic())
            logger.info("[ollama_guard] Ollama ready in %.1fs ✓", elapsed)
            print(f"\n[Jarvis] ✅ Ollama запущена!", flush=True)
            return True
        dots += 1
        print(f"\r[Jarvis] Жду Ollama{'.' * (dots % 4):<4}", end="", flush=True)
        time.sleep(_POLL_INTERVAL)

    logger.error("[ollama_guard] Ollama did not respond in %ds", _STARTUP_WAIT)
    print(
        f"\n[Jarvis] ❌ Ollama не ответила за {_STARTUP_WAIT}с. "
        "Запусти `ollama serve` вручную и перезапусти Jarvis.",
        flush=True,
    )
    return False
