import json
import threading
from pathlib import Path

from core.config import MAX_HISTORY
from core.paths import ROOT

_HISTORY_PATH = ROOT / "data" / "session.json"
_history: list[dict] = []
_lock = threading.Lock()


def _load_from_disk() -> None:
    """Загружает историю с диска при старте."""
    global _history
    try:
        if _HISTORY_PATH.exists():
            data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _history = data
                print(f"[history] Загружено {len(_history)} сообщений из {_HISTORY_PATH}")
    except Exception as e:
        print(f"[history] Не удалось загрузить сессию: {e}")
        _history = []


def _save_to_disk() -> None:
    """Сохраняет историю на диск (вызывается под _lock)."""
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_PATH.write_text(
            json.dumps(_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[history] Не удалось сохранить сессию: {e}")


# Загружаем при импорте модуля
_load_from_disk()


def append(role: str, content: str) -> None:
    with _lock:
        _history.append({"role": role, "content": content})
        max_msgs = MAX_HISTORY * 2
        if len(_history) > max_msgs:
            excess = len(_history) - max_msgs
            excess = excess + (excess % 2)  # только чётное — не рвём пары
            del _history[:excess]
        _save_to_disk()


def snapshot() -> list[dict]:
    with _lock:
        return list(_history)


def clear() -> None:
    """Сброс истории — и RAM, и файл."""
    with _lock:
        _history.clear()
        _save_to_disk()


def lock():
    return _lock
