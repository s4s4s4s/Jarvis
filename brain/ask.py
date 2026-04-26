from __future__ import annotations

import json
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from brain.client import chat, MODEL_ROUTER, MODEL_FAST
from brain.prompts import ROUTER_SYSTEM, TOOL_FORMAT_SYSTEM
from brain import history as hist
from brain.logger import log_route

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
                self._answer = f"\u041e\u0448\u0438\u0431\u043a\u0430: {e}"
            self._future = None
        return self._answer


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
                    f"\u0417\u0430\u043f\u0440\u043e\u0441: {text}\n\n"
                    f"\u0414\u0430\u043d\u043d\u044b\u0435 \u0438\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442\u0430 ({route_data['tool']}):\n"
                    f"{json.dumps(result.data, ensure_ascii=False, indent=2)}"
                )},
            ]
            return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 4096})
        return f"\u0421\u044d\u0440, \u0438\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442 \u0432\u0435\u0440\u043d\u0443\u043b \u043e\u0448\u0438\u0431\u043a\u0443: {result.error}"

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
    import time
    history = hist.snapshot()
    hist.append("user", text)

    route_data = _route(text)
    filler = route_data.get("filler", "")
    result = AskResult(filler=filler)

    def _run() -> str:
        t0 = time.monotonic()
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
        # Async memory extraction — never blocks the answer
        try:
            from tools.memory import extract_and_save_async
            extract_and_save_async(text, answer)
        except Exception:
            pass
        return answer

    result._future = _executor.submit(_run)
    return result
