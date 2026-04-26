from __future__ import annotations

import atexit
import json
import time
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from brain.client import chat, MODEL_ROUTER, MODEL_FAST
from brain.prompts import ROUTER_SYSTEM, TOOL_FORMAT_SYSTEM
from brain import history as hist
from brain.logger import log_route

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")
atexit.register(lambda: _executor.shutdown(wait=False, cancel_futures=True))


@dataclass
class AskResult:
    filler: str = ""
    _future: Future | None = field(default=None, repr=False)
    _answer: str = field(default="", repr=False)

    def get_answer(self, timeout: float = 120.0) -> str:
        if self._future is not None:
            try:
                self._answer = self._future.result(timeout=timeout)
            except Exception as e:
                self._answer = f"Ошибка: {e}"
            self._future = None
        return self._answer


# Эвристика filler — отдаётся немедленно, без ожидания роутера
_QUICK_FILLERS = [
    "Позвольте уточнить...",
    "Одну секунду...",
    "Съезжу и проверю.",
    "Анализирую.",
    "Дайте подумаю.",
]

_filler_idx = 0


def _quick_filler(text: str) -> str:
    """Быстрый filler по тексту запроса, не требует LLM."""
    global _filler_idx
    t = text.lower()
    if any(w in t for w in ("погода", "температура")):
        return "Смотрю текущие данные."
    if any(w in t for w in ("курс", "валюта", "доллар", "рубль")):
        return "Проверяю курсы."
    if any(w in t for w in ("крипто", "биткоин", "эфир")):
        return "Смотрю цены."
    if any(w in t for w in ("время", "час", "сколько")):
        return "Сейчас скажу."
    if any(w in t for w in ("найди", "поищи", "ищи", "гугл")):
        return "Ищу в сети."
    if any(w in t for w in ("объясни", "почему", "как", "расскажи")):
        return "Дайте подумаю."
    f = _QUICK_FILLERS[_filler_idx % len(_QUICK_FILLERS)]
    _filler_idx += 1
    return f


def _route(text: str) -> dict[str, Any]:
    msgs = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = chat(MODEL_ROUTER, msgs, options={"temperature": 0.0, "num_ctx": 4096})
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        data = {}
    return {
        "route": data.get("route", "chat"),
        "tool": data.get("tool"),
        "tool_args": data.get("tool_args") or {},
        "confidence": data.get("confidence", 0.0),
        "filler": data.get("filler") or "",
        "reason": data.get("reason") or "",
    }


def _dispatch(route_data: dict[str, Any], text: str, history: list[dict]) -> str:
    route = route_data["route"]

    if route == "tool":
        from brain.agents.tool_agent import tool_agent
        result = tool_agent(route_data["tool"], route_data["tool_args"])
        if result.ok:
            msgs = [
                {"role": "system", "content": TOOL_FORMAT_SYSTEM},
                {"role": "user", "content": (
                    f"Запрос: {text}\n\n"
                    f"Данные инструмента ({route_data['tool']}):\n"
                    f"{json.dumps(result.data, ensure_ascii=False, indent=2)}"
                )},
            ]
            return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 4096})
        return f"Сэр, инструмент вернул ошибку: {result.error}"

    if route == "web":
        from brain.agents.web_agent import run as web_run
        return web_run(text, history)

    if route == "deep":
        from brain.agents.deep import run as deep_run
        return deep_run(text, history)

    if route == "memory":
        from brain.agents.memory_agent import run as memory_run
        return memory_run(text, history)

    from brain.agents.chat import run as chat_run
    return chat_run(text, history)


def ask_llm(text: str) -> AskResult:
    history = hist.snapshot()
    hist.append("user", text)

    # filler отдаётся немедленно — без ожидания роутера
    filler = _quick_filler(text)
    result = AskResult(filler=filler)

    def _run() -> str:
        t0 = time.monotonic()
        # Роутер запускается внутри executor — не блокирует основной поток
        route_data = _route(text)
        answer = _dispatch(route_data, text, history)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        hist.append("assistant", answer)
        log_route(
            text=text,
            route=route_data["route"],
            tool=route_data.get("tool"),
            confidence=route_data.get("confidence", 0.0),
            reason=route_data.get("reason", ""),
            answer_ms=elapsed_ms,
        )
        try:
            from tools.memory import extract_and_save_async
            extract_and_save_async(text, answer)
        except Exception:
            pass
        return answer

    result._future = _executor.submit(_run)
    return result
