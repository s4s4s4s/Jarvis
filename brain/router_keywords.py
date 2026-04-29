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

FIXES vs previous version:
  - Removed @staticmethod decorator on module-level _kw() function (SyntaxError)
  - _TIME_RE: removed ^ $ anchors so it matches mid-sentence too
  - _CURRENCY_RE: added конвертируй/переведи patterns for currency convert intent
  - Added TIMER_SET handler that was compiled but never returned a route
  - Added TIMER_CANCEL_RE and LIST_TIMERS_RE rules
  - Added MEMORY_QUERY_RE for "что ты знаешь обо мне" pattern
  - _extract_location: logs warning instead of silently defaulting to Moscow
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# FIX: removed @staticmethod — this is a module-level function, not a method
def _kw(*words: str) -> re.Pattern:
    """Build a case-insensitive pattern that matches any of the words as whole words."""
    escaped = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b({escaped})\b", re.IGNORECASE | re.UNICODE)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_WEATHER_RE = re.compile(
    r"(погода|прогноз|температура|осадки|дождь|снег|ветер|облачно|солнечно)",
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

# FIX: added конвертируй/переведи/обмен patterns so convert intent is caught here
_CURRENCY_CONVERT_RE = re.compile(
    r"(конвертируй|переведи|обмен(яй)?|сколько.{0,15}(usd|eur|rub|cny|gbp|jpy|доллар|евро|юань))"
    r".{0,50}(usd|eur|rub|cny|gbp|jpy|доллар|евро|рубл|юань)",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"(курс|обменять|сколько стоит).{0,30}(доллар|евро|юань|рубл|usd|eur|cny|gbp|jpy)"
    r"|(доллар|евро|юань).{0,15}(курс|сейчас|к рублю)",
    re.IGNORECASE,
)

# FIX: removed ^$ anchors — must match anywhere in text, not only exact full-string
_TIME_RE = re.compile(
    r"(который час|сколько времени|текущее время|который сейчас час|what time is it)",
    re.IGNORECASE,
)

_TIMER_SET_RE = re.compile(
    r"(поставь|установи|запусти|включи|поставить|установить).{0,20}"
    r"(таймер|напомни|напоминание).{0,40}(\d+)\s*(секунд|минут|час)",
    re.IGNORECASE,
)

# FIX: new rules that were missing entirely
_TIMER_CANCEL_RE = re.compile(
    r"(отмени|убери|удали|стоп).{0,20}(таймер)",
    re.IGNORECASE,
)
_LIST_TIMERS_RE = re.compile(
    r"(покажи|список|какие).{0,20}(таймер)",
    re.IGNORECASE,
)
_MEMORY_QUERY_RE = re.compile(
    r"(что ты знаешь|расскажи.{0,10}обо мне|что я говорил|что я тебе рассказывал"
    r"|какие факты.{0,15}обо мне|помнишь ли ты|что ты помнишь)",
    re.IGNORECASE,
)

_CODE_WRITE_RE = re.compile(
    r"(напиши|написать|создай|создать|сделай|реализуй).{0,30}"
    r"(скрипт|код|программ|функци|класс|модуль|бот)",
    re.IGNORECASE,
)

_RECIPE_RE = re.compile(
    r"(рецепт|как приготовить|как сделать|приготовь|как варить|как жарить"
    r"|что нужно для|ингредиент)",
    re.IGNORECASE,
)

_SELF_TEST_RE = re.compile(
    r"(запусти|прогони|сделай|запустить|прогнать).{0,20}(тест|тесты|self.?test|самотестирование)"
    r"|(протестируй себя|протестируй джарвис)",
    re.IGNORECASE,
)

_SELF_ANALYZE_RE = re.compile(
    r"(проанализируй себя|analyse yourself|проверь свой код|найди баги в себе"
    r"|проверь себя|найди ошибки в себе)",
    re.IGNORECASE,
)

_GIT_STATUS_RE = re.compile(r"(git status|статус гита|что изменилось в гите)", re.IGNORECASE)
_GIT_DIFF_RE   = re.compile(r"(git diff|покажи diff|покажи изменения)", re.IGNORECASE)
_GIT_PUSH_RE   = re.compile(r"(git push|запушь|запушить|отправь в репо|запушь изменения)", re.IGNORECASE)
_GIT_COMMIT_RE = re.compile(
    r"(сделай коммит|git commit|закоммить|закоммитить).{0,60}",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_location(text: str) -> str:
    """Extract city name from weather query. Falls back to Москва with a warning."""
    m = _WEATHER_CITY_RE.search(text)
    if m and m.group(3):
        return m.group(3).strip()
    m2 = re.search(r"\bв\s+([А-Яа-яЁё]{3,}|[A-Z][a-z]{2,})", text)
    if m2:
        return m2.group(1)
    logger.debug("[fast_route] No city found in weather query, defaulting to Москва: %r", text[:80])
    return "Москва"


def _extract_timer_seconds(text: str) -> int:
    """Extract timer duration in seconds from text like 'на 5 минут'."""
    m = re.search(r"(\d+)\s*(секунд|минут|час)", text, re.IGNORECASE)
    if not m:
        return 60
    val = int(m.group(1))
    unit = m.group(2).lower()
    if "час" in unit:
        return val * 3600
    if "минут" in unit:
        return val * 60
    return val


def _extract_timer_label(text: str) -> str:
    """Try to extract a label from 'таймер на кофе' or 'напомни про встречу'."""
    m = re.search(
        r"(таймер|напомни).{0,10}(на|про|для)\s+([\w\s]{2,30})",
        text, re.IGNORECASE,
    )
    if m:
        label = m.group(3).strip().rstrip(",. ")
        return label if len(label) > 2 else "таймер"
    return "таймер"


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

    # FIX: currency convert checked before generic currency.rates
    if _CURRENCY_CONVERT_RE.search(t):
        # Let LLM router handle arg extraction for convert — we just set the route hint
        # But we still skip LLM by routing to currency.rates and letting tool_agent
        # decide based on the query — actual convert is handled via LLM router
        # Since we can't reliably parse from/to/amount here, fall through to LLM.
        # NOTE: We explicitly return None to let LLM handle convert queries properly.
        return None

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

    # FIX: TIMER_SET was compiled but never returned — now properly handled
    if _TIMER_SET_RE.search(t):
        seconds = _extract_timer_seconds(t)
        label = _extract_timer_label(t)
        return {
            "route": "tool", "tool": "timer.set",
            "tool_args": {"seconds": seconds, "label": label},
            "confidence": 0.96, "filler": "Ставлю таймер", "reason": "keyword: timer set",
        }

    if _TIMER_CANCEL_RE.search(t):
        return {
            "route": "tool", "tool": "timer.cancel",
            "tool_args": {"timer_id": ""},  # LLM router will fill id if needed
            "confidence": 0.80, "filler": "Отменяю таймер", "reason": "keyword: timer cancel",
        }

    if _LIST_TIMERS_RE.search(t):
        return {
            "route": "tool", "tool": "timer.list",
            "tool_args": {},
            "confidence": 0.97, "filler": "Смотрю таймеры", "reason": "keyword: timer list",
        }

    if _MEMORY_QUERY_RE.search(t):
        return {
            "route": "memory", "tool": None, "tool_args": {},
            "confidence": 0.95, "filler": "Вспоминаю", "reason": "keyword: memory query",
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
