# brain/explicit_feedback.py
"""
Явная обратная связь от пользователя голосом.

Триггеры (определяются роутером):
  route="feedback", tool="feedback.correct"  → последний ответ был верным
  route="feedback", tool="feedback.wrong"    → последний ответ был неверным

Хранит ID последней записи feedback_store, чтобы пометить её.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_last_record_id: str | None = None
_lock = threading.Lock()


def set_last_id(record_id: str) -> None:
    global _last_record_id
    with _lock:
        _last_record_id = record_id


def get_last_id() -> str | None:
    with _lock:
        return _last_record_id


def user_says_correct() -> str:
    rid = get_last_id()
    if rid:
        try:
            from brain.feedback_store import mark_verified
            mark_verified(rid, correct=True)
            logger.info(f"[feedback] Verified correct: {rid}")
        except Exception as e:
            logger.error(f"[feedback] mark_verified error: {e}")
        return "Отлично, запомню что так правильно."
    return "Хорошо."


def user_says_wrong() -> str:
    rid = get_last_id()
    if rid:
        try:
            from brain.feedback_store import mark_verified
            mark_verified(rid, correct=False)
            logger.info(f"[feedback] Verified wrong: {rid}")
            # Немедленно удаляем авто-пример если он есть
            threading.Thread(
                target=_trigger_immediate_removal,
                args=(rid,),
                daemon=True,
                name="jarvis-learning-trigger",
            ).start()
        except Exception as e:
            logger.error(f"[feedback] user_says_wrong error: {e}")
        return "Понял, запомню ошибку. В следующий раз буду точнее."
    return "Хорошо, учту."


def _trigger_immediate_removal(record_id: str) -> None:
    try:
        from brain.learning_loop import remove_failed_example
        remove_failed_example(record_id)
    except Exception as e:
        logger.error(f"[feedback] immediate removal error: {e}")
