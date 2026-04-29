# brain/explicit_feedback.py
"""
Явная обратная связь от пользователя голосом.

Подключается через маршрут feedback в ROUTER_SYSTEM.
Хранит ID последней записи feedback_store, чтобы пометить её верифицированной.
"""
from __future__ import annotations

import logging
import threading

from brain.feedback_store import mark_verified

logger = logging.getLogger(__name__)

_last_record_id: str | None = None
_lock = threading.Lock()


def set_last_id(record_id: str) -> None:
    """Вызывается из ask.py после каждого ответа."""
    global _last_record_id
    with _lock:
        _last_record_id = record_id


def get_last_id() -> str | None:
    with _lock:
        return _last_record_id


def user_says_correct() -> str:
    rid = get_last_id()
    if rid:
        mark_verified(rid, correct=True)
        logger.info(f"[explicit_feedback] Верифицировано как SUCCESS: {rid}")
        return "Отлично, запомню что так правильно."
    return "Хорошо."


def user_says_wrong() -> str:
    rid = get_last_id()
    if rid:
        mark_verified(rid, correct=False)
        logger.info(f"[explicit_feedback] Верифицировано как FAILURE: {rid}")
        threading.Thread(
            target=_trigger_learning,
            args=(rid,),
            daemon=True,
            name="jarvis-learning-trigger",
        ).start()
        return "Понял, запомню ошибку. В следующий раз сделаю лучше."
    return "Понял, учту."


def run(tool: str) -> str:
    """Точка входа из _dispatch в ask.py."""
    if tool == "feedback.correct":
        return user_says_correct()
    if tool == "feedback.wrong":
        return user_says_wrong()
    return "Хорошо."


def _trigger_learning(record_id: str) -> None:
    try:
        from brain.learning_loop import add_failure_example
        add_failure_example(record_id)
    except Exception as e:
        logger.error(f"[explicit_feedback] Ошибка немедленного обучения: {e}")
