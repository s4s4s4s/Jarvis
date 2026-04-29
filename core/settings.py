# core/settings.py
"""
Persistent user settings stored in data/settings.json.
Used at runtime instead of hardcoded values in core/config.py.

FIX: renamed public `set()` to `set_value()` to avoid shadowing Python builtin `set`.
The old name `set` is kept as an alias for backward compatibility.
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

_SETTINGS_PATH = Path(__file__).parent.parent / "data" / "settings.json"
_lock = threading.Lock()

_DEFAULTS: dict = {
    "mic_device": None,   # None = system default; int = sounddevice device index
}


def _load() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            with _SETTINGS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {**_DEFAULTS, **data}
    except Exception:
        pass
    return dict(_DEFAULTS)


def _save(data: dict) -> None:
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _SETTINGS_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[settings] save error: {e}")


_cache: dict | None = None


def get(key: str, default=None):
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load()
        return _cache.get(key, default)


def set_value(key: str, value) -> None:  # noqa: A001
    """Save a setting. Preferred name to avoid shadowing builtin `set`."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load()
        _cache[key] = value
        _save(_cache)


# Backward-compat alias — existing call sites that use settings.set() still work
set = set_value  # noqa: A001


def all_settings() -> dict:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load()
        return dict(_cache)
