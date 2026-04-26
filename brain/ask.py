from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from brain.client import chat, MODEL_ROUTER
from brain.prompts import ROUTER_SYSTEM
from brain import history as hist

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")


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


def _route(text: str) -> dict[str, Any]:
    msgs = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = chat(MODEL_ROUTER, msgs, options={"temperature": 0.0, "num_ctx": 4096})
    try:
        raw = raw.strip()
        # strip possible markdown code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
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
            # Format tool result via fast LLM for natural language response
            from brain.client import MODEL_FAST
            from brain.prompts import TOOL_FORMAT_SYSTEM
            msgs = [
                {"role": "system", "content": TOOL_FORMAT_SYSTEM},
                {"role": "user", "content": f"Запрос: {text}\n\nДанные инструмента ({route_data['tool']}):\n{json.dumps(result.data, ensure_ascii=False, indent=2)}"},
            ]
            return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 4096})
        else:
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

    # default: chat
    from brain.agents.chat import run as chat_run
    return chat_run(text, history)


def ask_llm(text: str) -> AskResult:
    history = hist.snapshot()
    hist.append("user", text)

    route_data = _route(text)
    filler = route_data.get("filler", "")

    result = AskResult(filler=filler)

    def _run() -> str:
        answer = _dispatch(route_data, text, history)
        hist.append("assistant", answer)
        return answer

    result._future = _executor.submit(_run)
    return result
