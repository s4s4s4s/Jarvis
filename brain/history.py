import threading

from core.config import MAX_HISTORY

_history: list[dict] = []
_lock = threading.Lock()


def append(role: str, content: str) -> None:
    with _lock:
        _history.append({"role": role, "content": content})
        max_msgs = MAX_HISTORY * 2  # каждый ход = 2 сообщения (user + assistant)
        if len(_history) > max_msgs:
            del _history[: len(_history) - max_msgs]


def snapshot() -> list[dict]:
    with _lock:
        return list(_history)


def lock():
    return _lock
