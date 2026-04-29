# tools/system/clipboard.py
"""
Работа с буфером обмена Windows для Jarvis.

Инструменты: clipboard.get, clipboard.set
"""
from __future__ import annotations

import subprocess
import sys

_PLATFORM = sys.platform


def get_clipboard() -> dict:
    """
    Возвращает текущее содержимое буфера обмена.
    Возвращает: {text, length}
    """
    try:
        if _PLATFORM == "win32":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            text = result.stdout.strip()
        else:
            # Linux/Mac fallback
            result = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                                    capture_output=True, text=True, timeout=5)
            text = result.stdout
        return {"text": text, "length": len(text)}
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения буфера обмена: {e}")


def set_clipboard(text: str) -> dict:
    """
    Записывает текст в буфер обмена.
    Возвращает: {text, length, ok}
    """
    try:
        if _PLATFORM == "win32":
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Set-Clipboard -Value '{text.replace(chr(39), chr(96))}'"],
                timeout=5, check=True,
            )
        else:
            proc = subprocess.Popen(["xclip", "-selection", "clipboard"],
                                    stdin=subprocess.PIPE)
            proc.communicate(input=text.encode())
        return {"text": text, "length": len(text), "ok": True}
    except Exception as e:
        raise RuntimeError(f"Ошибка записи в буфер обмена: {e}")
