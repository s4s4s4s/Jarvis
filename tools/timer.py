# tools/timer.py
"""
tools/timer.py — простые голосовые таймеры и напоминания.

Использование:
    set_timer(seconds=300, label="кофе")  -> запускает фоновый Thread
    list_timers()                          -> список активных таймеров
    cancel_timer(timer_id)                 -> отменяет таймер

Когда таймер срабатывает — он вызывает _fire_callback, который по умолчанию
печатает сообщение. voice/assistant.py подменяет его на say() при старте.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional


@dataclass
class Timer:
    id: str
    label: str
    seconds: int
    fire_at: datetime
    _thread: threading.Timer = field(repr=False, default=None)
    cancelled: bool = False


_timers: dict[str, Timer] = {}
_timers_lock = threading.Lock()

# Callback, вызываемый при срабатывании таймера.
# По умолчанию — print. voice/assistant.py может подменить на say().
_fire_callback: Callable[[str], None] = print


def set_fire_callback(cb: Callable[[str], None]) -> None:
    """Подменить callback (вызывается из voice/assistant.py)."""
    global _fire_callback
    _fire_callback = cb


def _on_fire(timer_id: str, label: str) -> None:
    with _timers_lock:
        t = _timers.pop(timer_id, None)
        if t is None or t.cancelled:
            return
    msg = f"Сэр, таймер '{label}' сработал!"
    try:
        _fire_callback(msg)
    except Exception as e:
        print(f"[timer] Ошибка callback: {e}")


def set_timer(seconds: int, label: str = "таймер") -> dict:
    """
    Устанавливает таймер на seconds секунд.
    Возвращает dict с id, label, seconds, fire_at.
    """
    if seconds <= 0:
        raise ValueError("seconds должен быть > 0")
    if seconds > 86400:
        raise ValueError("Максимальное время таймера — 24 часа")

    timer_id = uuid.uuid4().hex[:8]
    fire_at = datetime.now() + timedelta(seconds=seconds)

    t = threading.Timer(seconds, _on_fire, args=(timer_id, label))
    t.daemon = True

    entry = Timer(
        id=timer_id,
        label=label,
        seconds=seconds,
        fire_at=fire_at,
        _thread=t,
    )
    with _timers_lock:
        _timers[timer_id] = entry
    t.start()
    print(f"[timer] Запущен '{label}' на {seconds}с (id={timer_id})")
    return {
        "id": timer_id,
        "label": label,
        "seconds": seconds,
        "fire_at": fire_at.strftime("%H:%M:%S"),
    }


def list_timers() -> list[dict]:
    """Возвращает список активных таймеров."""
    with _timers_lock:
        now = datetime.now()
        result = []
        for t in list(_timers.values()):
            remaining = max(0, int((t.fire_at - now).total_seconds()))
            result.append({
                "id": t.id,
                "label": t.label,
                "remaining_seconds": remaining,
                "fire_at": t.fire_at.strftime("%H:%M:%S"),
            })
        return result


def cancel_timer(timer_id: str) -> bool:
    """Отменяет таймер по id. Возвращает True если нашёл и отменил."""
    with _timers_lock:
        t = _timers.pop(timer_id, None)
    if t is None:
        return False
    t.cancelled = True
    if t._thread:
        t._thread.cancel()
    print(f"[timer] Отменён '{t.label}' (id={timer_id})")
    return True


def cancel_all() -> int:
    """Отменяет все таймеры. Возвращает количество отменённых."""
    with _timers_lock:
        ids = list(_timers.keys())
    count = sum(1 for tid in ids if cancel_timer(tid))
    return count
