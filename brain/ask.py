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
                # FIX (аудит 6): использовать LLM для генерации ошибки вместо хардкода
                self._answer = _format_tool_error("получение ответа", "get_answer", str(e))
            self._future = None
        return self._answer


# FIX (аудит 6): полностью убраны захардкоженные филлеры и keyword-based
# роутинг филлеров (_QUICK_FILLERS, _SMALL_TALK_PATTERNS, _quick_filler).
# Принципиальный архитектурный сдвиг: Jarvis — ИИ-агент, не чат-бот.
# Все фразы, которые произносит ассистент, должна генерировать LLM.
# Филлер теперь приходит из роутера в JSON-поле "filler" (см. ROUTER_SYSTEM).
# Это даёт контекстно-уместные филлеры ценой ~500-1500мс на инференс роутера.


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


def _format_tool_error(text: str, tool_name: str | None, error: str) -> str:
    """LLM генерирует естественный ответ об ошибке инструмента.

    FIX (аудит 6): раньше тут был хардкод "Сэр, инструмент вернул ошибку: ...".
    Теперь LLM сама формулирует ответ пользователю на основе контекста ошибки.
    """
    msgs = [
        {"role": "system", "content": TOOL_FORMAT_SYSTEM},
        {"role": "user", "content": (
            f"Запрос пользователя: {text}\n\n"
            f"Инструмент ({tool_name}) завершился с ошибкой:\n{error}\n\n"
            f"Сообщи пользователю, что не удалось выполнить запрос, "
            f"кратко объясни причину естественным языком."
        )},
    ]
    try:
        return chat(MODEL_FAST, msgs, options={"temperature": 0.3, "num_ctx": 4096})
    except Exception as e:
        # Fallback на хардкод только если LLM недоступна
        # FIX (аудит 6): добавлен логгинг для молчаливых исключений
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"LLM error in _format_tool_error for tool '{tool_name}': {e}")
        return f"Сэр, инструмент вернул ошибку: {error}"


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
        # FIX (аудит 6): ошибку формулирует LLM, а не хардкод
        return _format_tool_error(text, route_data.get("tool"), result.error or "неизвестная ошибка")

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
    # FIX: snapshot берём ДО append(user) — это правильный контекст для агентов.
    # append(user) тоже до executor, чтобы при параллельных вызовах (прерывание)
    # порядок в истории был user1 → user2, а не user1 → assistant1 → user2.
    history = hist.snapshot()
    hist.append("user", text)

    # FIX (аудит 6): роутер вызывается СИНХРОННО здесь, чтобы получить
    # контекстно-уместный филлер вместе с маршрутом. Раньше филлер брался
    # из keyword-based _quick_filler() (~0мс, но тупо). Теперь — от LLM
    # (~500-1500мс, но умно). Это согласуется с философией "ИИ-агент,
    # не чат-бот": все фразы генерирует LLM. Если роутер упал —
    # AskResult вернёт пустой filler и chat-маршрут по умолчанию.
    t_route0 = time.monotonic()
    try:
        route_data = _route(text)
    except Exception as e:
        # Ollama упала / сетевой сбой — отдаём в воркер сам факт ошибки,
        # чтобы LLM-цепочка тоже отвалилась с понятным сообщением.
        route_data = {
            "route": "chat",
            "tool": None,
            "tool_args": {},
            "confidence": 0.0,
            "filler": "",
            "reason": f"router error: {e}",
        }
    route_ms = int((time.monotonic() - t_route0) * 1000)

    result = AskResult(filler=route_data.get("filler", ""))

    def _run() -> str:
        t0 = time.monotonic()
        answer = _dispatch(route_data, text, history)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        # assistant-ответ добавляется после получения, не в главном потоке —
        # это сохраняет правильное чередование при быстрых последовательных вопросах
        hist.append("assistant", answer)
        log_route(
            text=text,
            route=route_data["route"],
            tool=route_data.get("tool"),
            confidence=route_data.get("confidence", 0.0),
            reason=route_data.get("reason", ""),
            answer_ms=elapsed_ms + route_ms,
        )
        try:
            from tools.memory import extract_and_save_async
            extract_and_save_async(text, answer)
        except Exception as e:
            # FIX (аудит 6): добавлен логгинг для молчаливого исключения
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Memory extraction failed: {e}")
        return answer

    result._future = _executor.submit(_run)
    return result
