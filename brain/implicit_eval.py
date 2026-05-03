# brain/implicit_eval.py
"""
Неявная (автоматическая) оценка качества ответа.

Логика:
1. Если инструмент вернул ошибку → failure
2. Если ответ содержит признаки провала при route=tool/web → failure
3. Если confidence < порога → unknown
4. Иначе → success (предварительно)

Это не истина — это стартовая оценка. Пользователь может её переопределить голосом.

fix F3: исправлен фолс-позитив failure для паттернов «ошибка»/«не удалось» в контекстных
фразах (напр.: "чтобы избежать ошибки", "без ошибок").
Теперь проверяем паттерн на 10-символьном слове (word boundary via \\b или
более точные формулировки). Слова без глагола действия ("ошибка" как существительное
без предиката провала) вычеркнуты из списка.
"""
from __future__ import annotations

import re

# fix F3: список паттернов переработан.
#
# Удалены одинарные r"ошибка" и r"не удалось" — они ловили сообщения вида
# "чтобы избежать ошибки", "без ошибок", "не удалось добавить… и это правильно".
#
# Оставлены только чёткие индикаторы фактического провала:
# - "произошла ошибка" / "возникла ошибка" (глагол + ошибка)
# - "не смог" / "не смогла" (полная форма, без усечения)
# - "error" / "exception" / "traceback" (технические маркеры, безамбигуальные)
# - "недоступен" / "не нашёл" / "не могу найти" (фактический отказ)
# - "поиск не дал" (четкая формулировка, не имеет безвредных трактовок)
# - "инструмент.*ошибк" (сигнал от tool_agent о провале инструмента)
_FAILURE_PATTERNS = [
    r"(?:произошла|возникла)\s+ошибка",   # "произошла ошибка" / "возникла ошибка"
    r"не\s+смог(?:ла?)\b",                     # "не смог" / "не смогла"
    r"\bnedostupen\b|\u043dедоступен\b",           # "недоступен"
    r"не\s+нашёл\b",                           # "не нашёл"
    r"не\s+могу\s+найти\b",                 # "не могу найти"
    r"поиск\s+не\s+дал\b",                    # "поиск не дал"
    r"инструмент.*ошибк",                  # "инструмент.*ошибка"
    r"\berror\b",                              # "техническое error"
    r"\bexception\b",                          # "техническое exception"
    r"\btraceback\b",                          # Python traceback
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
