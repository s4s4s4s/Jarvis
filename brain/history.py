# C:\jarvis\brain\history.py
import threading

_history: list[dict] = []
_lock = threading.Lock()
MAX_TURNS = 20


def append(role: str, content: str) -> None:
    with _lock:
        _history.append({"role": role, "content": content})
        if len(_history) > MAX_TURNS * 2:
            del _history[: len(_history) - MAX_TURNS * 2]


def snapshot() -> list[dict]:
    with _lock:
        return list(_history)


def lock():
    return _lock
# === end of file: brain/history.py ===