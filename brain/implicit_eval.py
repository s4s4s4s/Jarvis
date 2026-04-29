# brain/implicit_eval.py
"""
Автоматическая (неявная) оценка качества ответа Jarvis.

Это не истина — это стартовая оценка на основе эвристик.
Пользователь может переопределить её через явную обратную связь.
"""
from __future__ import annotations

import re

_FAILURE_PATTERNS = [
    r"не удалось",
    r"ошибка",
    r"не смог",
    r"недоступен",
    r"не нашёл",
    r"не могу найти",
    r"поиск не дал",
    r"инструмент.*ошибк",
    r"сервис недоступен",
    r"нет данных",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FAILURE_PATTERNS]

MIN_CONFIDENCE_FOR_SUCCESS = 0.65


def evaluate(
    route: str,
    tool: str | None,
    confidence: float,
    answer: str,
    tool_ok: bool = True,
) -> tuple[str, str]:
    """
    Возвращает (outcome, reason).
    outcome: "success" | "failure" | "unknown"
    """
    if not tool_ok:
        return "failure", "tool_error"

    for pattern in _COMPILED:
        if pattern.search(answer):
            if route in ("tool", "web"):
                return "failure", "answer_contains_failure_pattern"

    if confidence < MIN_CONFIDENCE_FOR_SUCCESS:
        return "unknown", f"low_confidence={confidence:.2f}"

    return "success", ""
