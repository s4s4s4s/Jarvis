import json
import threading
from pathlib import Path

from core.config import MAX_HISTORY
from core.paths import ROOT

_HISTORY_PATH = ROOT / "data" / "session.json"
_history: list[dict] = []
_lock = threading.Lock()


def _load_from_disk() -> None:
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
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_PATH.write_text(
            json.dumps(_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[history] Не удалось сохранить сессию: {e}")


_load_from_disk()


def append(role: str, content: str) -> None:
    with _lock:
        _history.append({"role": role, "content": content})
        max_msgs = MAX_HISTORY * 2
        if len(_history) > max_msgs:
            excess = len(_history) - max_msgs
            # FIX: правильное округление вверх до чётного числа.
            # Старая формула excess + (excess % 2) давала:
            #   excess=1 → 1+1=2 (удаляла лишнее сообщение)
            #   excess=2 → 2+0=2 (OK)
            # Правильно: ((excess + 1) // 2) * 2
            #   excess=1 → ((2)//2)*2 = 2 (округляем вверх до чётного — ОК,
            #              лучше удалить пару чем оставить сиротское сообщение)
            #   excess=2 → ((3)//2)*2 = 2 (OK)
            #   excess=3 → ((4)//2)*2 = 4 (OK)
            excess = ((excess + 1) // 2) * 2
            del _history[:excess]
        _save_to_disk()


def snapshot() -> list[dict]:
    with _lock:
        return list(_history)


def clear() -> None:
    with _lock:
        _history.clear()
        _save_to_disk()


def lock():
    return _lock
