"""brain/router_keywords.py

Level-1 router: instant keyword/regex matching before LLM.
Returns a partial route_data dict or None if not matched.

Priority: MANDATORY rules that should NEVER go to LLM.
Benefit: saves ~2s LLM call for 70-80% of common queries.

Usage (in brain/ask.py _route):
    from brain.router_keywords import fast_route
    data = fast_route(text)
    if data:
        return data
    # ... fall through to LLM router
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@staticmethod
def _kw(*words: str) -> re.Pattern:
    """Build a case-insensitive pattern that matches any of the words."""
    escaped = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b({escaped})\b", re.IGNORECASE | re.UNICODE)


_WEATHER_RE  = re.compile(
    r"(погода|прогноз|температура|осадки|дождь|снег|ветер|облачно|солнечно)"
    r".{0,60}(в\s+\w+|сегодня|завтра|сейчас|сейчас)?",
    re.IGNORECASE,
)
_WEATHER_CITY_RE = re.compile(
    r"(погода|прогноз|температура).{0,40}(в\s+([А-Яа-яЁёA-Za-z]{3,}))",
    re.IGNORECASE,
)

_CRYPTO_PRICE_RE = re.compile(
    r"(цена|стоимость|курс).{0,30}(btc|bitcoin|биткоин|eth|ethereum|эфириум|bnb|sol|solana|xrp)"
    r"|(btc|bitcoin|биткоин|eth|ethereum|bnb|sol|xrp).{0,20}(цена|стоимость|курс|сейчас)",
    re.IGNORECASE,
)

_CURRENCY_RE = re.compile(
    r"(курс|обменять|сколько стоит).{0,30}(доллар|евро|юань|рубл|usd|eur|cny|gbp|jpy)"
    r"|(доллар|евро|юань).{0,15}(курс|сейчас|к рублю)",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"^(который час|сколько времени|текущее время|который сейчас час|what time is it)[?!.]?$",
    re.IGNORECASE,
)

_TIMER_SET_RE = re.compile(
    r"(поставь|установи|запусти|включи).{0,20}(таймер|напомни).{0,30}(\d+)\s*(секунд|минут|час)",
    re.IGNORECASE,
)

_CODE_WRITE_RE = re.compile(
    r"(напиши|написать|создай|создать|сделай|реализуй).{0,30}(скрипт|код|программ|функци|класс|модуль)",
    re.IGNORECASE,
)

_RECIPE_RE = re.compile(
    r"(рецепт|как приготовить|как сделать|приготовь|как варить|как жарить)",
    re.IGNORECASE,
)

_SELF_TEST_RE = re.compile(
    r"(запусти|прогони|сделай).{0,20}(тест|тесты|self.?test|самотестирование)"
    r"|(протестируй себя)",
    re.IGNORECASE,
)

_SELF_ANALYZE_RE = re.compile(
    r"(проанализируй себя|analyse yourself|проверь свой код|найди баги в себе)",
    re.IGNORECASE,
)

_GIT_STATUS_RE  = re.compile(r"^(git status|статус гита|что изменилось в гите)[?!.]?$", re.IGNORECASE)
_GIT_DIFF_RE    = re.compile(r"^(git diff|покажи diff|покажи изменения)[?!.]?$", re.IGNORECASE)
_GIT_PUSH_RE    = re.compile(r"^(git push|запушь|запушить|отправь в репо)[?!.]?$", re.IGNORECASE)


def _extract_location(text: str) -> str:
    m = _WEATHER_CITY_RE.search(text)
    if m and m.group(3):
        return m.group(3).strip()
    # Generic: find word after 'в' near weather word
    m2 = re.search(r"\bв\s+([А-Яа-яЁё]{3,}|[A-Z][a-z]{2,})", text)
    if m2:
        return m2.group(1)
    return "Москва"  # default


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def fast_route(text: str) -> dict[str, Any] | None:
    """
    Try to route without LLM. Returns partial route_data dict or None.
    Covers the most common unambiguous cases.
    """
    t = text.strip()

    if _WEATHER_RE.search(t):
        loc = _extract_location(t)
        return {
            "route": "tool", "tool": "weather",
            "tool_args": {"location": loc, "language": "ru"},
            "confidence": 0.97, "filler": "Смотрю погоду", "reason": "keyword: weather",
        }

    if _CRYPTO_PRICE_RE.search(t):
        # Extract coin name
        m = re.search(
            r"(btc|bitcoin|биткоин|eth|ethereum|эфириум|bnb|sol|solana|xrp)",
            t, re.IGNORECASE,
        )
        coin_map = {
            "btc": "bitcoin", "bitcoin": "bitcoin", "биткоин": "bitcoin",
            "eth": "ethereum", "ethereum": "ethereum", "эфириум": "ethereum",
            "bnb": "binancecoin", "sol": "solana", "solana": "solana", "xrp": "ripple",
        }
        coin_id = coin_map.get((m.group(1) if m else "bitcoin").lower(), "bitcoin")
        return {
            "route": "tool", "tool": "crypto.price",
            "tool_args": {"ids": [coin_id], "vs_currency": "usd"},
            "confidence": 0.97, "filler": "Смотрю цену", "reason": "keyword: crypto price",
        }

    if _CURRENCY_RE.search(t):
        return {
            "route": "tool", "tool": "currency.rates",
            "tool_args": {},
            "confidence": 0.95, "filler": "Смотрю курсы", "reason": "keyword: currency",
        }

    if _TIME_RE.search(t):
        return {
            "route": "tool", "tool": "time",
            "tool_args": {},
            "confidence": 0.99, "filler": "Смотрю время", "reason": "keyword: time",
        }

    if _RECIPE_RE.search(t):
        return {
            "route": "plan", "tool": None, "tool_args": {},
            "confidence": 0.93, "filler": "Готовлю рецепт", "reason": "keyword: recipe",
        }

    if _CODE_WRITE_RE.search(t):
        return {
            "route": "code", "tool": None, "tool_args": {},
            "confidence": 0.92, "filler": "Пишу код", "reason": "keyword: write code",
        }

    if _SELF_TEST_RE.search(t):
        return {
            "route": "test", "tool": None, "tool_args": {},
            "confidence": 0.98, "filler": "Запускаю тесты", "reason": "keyword: self-test",
        }

    if _SELF_ANALYZE_RE.search(t):
        return {
            "route": "analyze", "tool": None, "tool_args": {},
            "confidence": 0.97, "filler": "Анализирую себя", "reason": "keyword: self-analyze",
        }

    if _GIT_STATUS_RE.search(t):
        return {
            "route": "tool", "tool": "git.status", "tool_args": {},
            "confidence": 0.99, "filler": "Смотрю статус", "reason": "keyword: git status",
        }

    if _GIT_DIFF_RE.search(t):
        return {
            "route": "tool", "tool": "git.diff", "tool_args": {"path": None},
            "confidence": 0.99, "filler": "Смотрю diff", "reason": "keyword: git diff",
        }

    if _GIT_PUSH_RE.search(t):
        return {
            "route": "tool", "tool": "git.push", "tool_args": {},
            "confidence": 0.99, "filler": "Пушу", "reason": "keyword: git push",
        }

    return None
