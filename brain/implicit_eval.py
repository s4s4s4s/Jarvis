# brain/implicit_eval.py
"""
Неявная (автоматическая) оценка качества ответа.

Логика:
1. Если инструмент вернул ошибку → failure
2. Если ответ содержит признаки провала при route=tool/web → failure
3. Если confidence < порога → unknown
4. Иначе → success (предварительно)

Это не истина — это стартовая оценка. Пользователь может её переопределить голосом.
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
    r"error",
    r"exception",
    r"traceback",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FAILURE_PATTERNS]

MIN_CONFIDENCE_FOR_SUCCESS = 0.60


def evaluate(
    route: str,
    tool: str | None,
    confidence: float,
    answer: str,
    tool_ok: bool = True,
) -> tuple[str, str]:
    """
    Returns (outcome, reason).
    outcome: "success" | "failure" | "unknown"
    reason:  краткое объяснение
    """
    # Явная ошибка инструмента
    if not tool_ok:
        return "failure", "tool_error"

    # Признаки провала в тексте ответа
    for pattern in _COMPILED:
        if pattern.search(answer):
            if route in ("tool", "web"):
                return "failure", "answer_contains_failure_pattern"

    # Низкая уверенность роутера
    if confidence < MIN_CONFIDENCE_FOR_SUCCESS:
        return "unknown", f"low_confidence={confidence:.2f}"

    return "success", ""
