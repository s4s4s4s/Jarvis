from __future__ import annotations

import atexit
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

from brain.client import chat, MODEL_ROUTER, MODEL_FAST
from brain.prompts import ROUTER_SYSTEM
from brain.prompts import TOOL_FORMAT_SYSTEM
from brain import history as hist
from brain.logger import log_route

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")
atexit.register(lambda: _executor.shutdown(wait=False, cancel_futures=True))

logger = logging.getLogger(__name__)

_FALLBACK_TIMEOUT_MSG = "Сэр, не удалось получить ответ вовремя."
_FALLBACK_ERROR = "Сэр, инструмент вернул ошибку: {error}"

_ROUTE_TIMEOUTS: dict[str, float] = {
    "test":    1800.0,
    "plan":     600.0,
    "code":     300.0,
    "deep":     300.0,
    "analyze":  600.0,   # self-analysis may take a while scanning all files
}
_DEFAULT_TIMEOUT = 120.0

_VOICE_TRUNCATE_TOOLS = {
    "auditor.self",
    "auditor.run",
    "file.read",
    "file.list",
    "git.diff",
    "git.status",
    "code.run",
    "code.run_file",
    "code.test",
    "self_test",
    "self_analysis",  # added: self-analysis produces long markdown output
}

_VOICE_MAX_CHARS = 220
_ROUTER_HISTORY_TURNS = 6

_tl = threading.local()


def report_progress(msg: str) -> None:
    """Call from any agent running inside ask_llm thread to push a live update."""
    cb = getattr(_tl, "progress_cb", None)
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass


@dataclass
class AskResult:
    filler: str = ""
    _text:        str   = field(default="", repr=False)
    _future:      Future | None = field(default=None, repr=False)
    _answer:      str   = field(default="", repr=False)
    _voice_reply: str   = field(default="", repr=False)
    _timeout:     float = field(default=_DEFAULT_TIMEOUT, repr=False)
    on_progress:  Callable[[str], None] | None = field(default=None, repr=False)

    def get_answer(self, timeout: float | None = None) -> str:
        effective = timeout if timeout is not None else self._timeout
        if self._future is not None:
            try:
                self._answer, self._voice_reply = self._future.result(timeout=effective)
            except FutureTimeoutError:
                logger.error(
                    "[ask] Future timed out after %.0fs for: %.80s",
                    effective, self._text,
                )
                self._answer = _FALLBACK_TIMEOUT_MSG
                self._voice_reply = _FALLBACK_TIMEOUT_MSG
            except Exception as e:
                logger.error(
                    "[ask] Future error (%s): %r  query=%.80s",
                    type(e).__name__, e, self._text,
                )
                self._answer = _FALLBACK_TIMEOUT_MSG
                self._voice_reply = _FALLBACK_TIMEOUT_MSG
            self._future = None
        return self._answer

    def get_voice_reply(self, timeout: float | None = None) -> str:
        if self._future is not None:
            self.get_answer(timeout=timeout)
        return self._voice_reply or self._answer


def _route(text: str, history: list[dict]) -> dict[str, Any]:
    recent = history[-_ROUTER_HISTORY_TURNS:] if history else []
    history_msgs = [
        {"role": m["role"], "content": m["content"][:800]}
        for m in recent
    ]
    msgs = [
        {"role": "system", "content": ROUTER_SYSTEM},
        *history_msgs,
        {"role": "user",   "content": text},
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
        "route":      data.get("route", "chat"),
        "tool":       data.get("tool"),
        "tool_args":  data.get("tool_args") or {},
        "confidence": data.get("confidence", 0.0),
        "filler":     data.get("filler") or "",
        "reason":     data.get("reason") or "",
    }


def _trim_for_voice(answer: str, tool_name: str | None) -> str:
    text = answer
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"\|[^\n]+", "", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if not text:
        text = re.sub(r"[`*#|\n]", " ", answer).strip()
        return text[:120].strip()

    if tool_name in _VOICE_TRUNCATE_TOOLS or len(text) > _VOICE_MAX_CHARS:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for i, sent in enumerate(sentences):
            candidate = (result + " " + sent).strip() if result else sent
            if i >= 2 or len(candidate) > _VOICE_MAX_CHARS:
                if not result:
                    result = candidate[:_VOICE_MAX_CHARS].rstrip() + "..."
                break
            result = candidate
        text = result or text[:_VOICE_MAX_CHARS].rstrip() + "..."

    if len(text) > _VOICE_MAX_CHARS:
        text = text[:_VOICE_MAX_CHARS].rstrip() + "..."

    return text


def _format_tool_error(text: str, tool_name: str | None, error: str) -> str:
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
        logger.error("LLM error in _format_tool_error for tool '%s': %s", tool_name, e)
        return _FALLBACK_ERROR.format(error=error)


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

    if route == "code":
        from brain.agents.code_agent import run as code_run
        return code_run(text, history)

    if route == "plan":
        from brain.agents.plan_agent import run as plan_run
        return plan_run(text, history)

    if route == "test":
        from brain.agents.self_test_agent import run as self_test_run
        return self_test_run(text, history)

    if route == "analyze":
        from brain.agents.self_analysis_agent import run as analyze_run
        return analyze_run(text, history)

    from brain.agents.chat import run as chat_run
    return chat_run(text, history)


def ask_llm(
    text: str,
    on_progress: Callable[[str], None] | None = None,
) -> AskResult:
    history = hist.snapshot()
    hist.append("user", text)

    t_route0 = time.monotonic()
    try:
        route_data = _route(text, history)
    except Exception as e:
        route_data = {
            "route":      "chat",
            "tool":       None,
            "tool_args":  {},
            "confidence": 0.0,
            "filler":     "",
            "reason":     f"router error: {e}",
        }
    route_ms = int((time.monotonic() - t_route0) * 1000)

    route = route_data["route"]
    timeout = _ROUTE_TIMEOUTS.get(route, _DEFAULT_TIMEOUT)
    result = AskResult(
        filler=route_data.get("filler", ""),
        _text=text,
        _timeout=timeout,
        on_progress=on_progress,
    )
    tool_name = route_data.get("tool")

    def _run() -> tuple[str, str]:
        _tl.progress_cb = on_progress
        try:
            t0 = time.monotonic()
            answer = _dispatch(route_data, text, history)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            hist.append("assistant", answer)
            log_route(
                text=text,
                route=route,
                tool=tool_name,
                confidence=route_data.get("confidence", 0.0),
                reason=route_data.get("reason", ""),
                answer_ms=elapsed_ms + route_ms,
            )
            try:
                from tools.memory import extract_and_save_async
                extract_and_save_async(text, answer)
            except Exception as e:
                logger.error("Memory extraction failed: %s", e)

            voice_reply = _trim_for_voice(answer, tool_name)
            logger.debug("[ask] voice_reply (%d chars): %s", len(voice_reply), voice_reply[:80])
            return answer, voice_reply
        finally:
            _tl.progress_cb = None

    result._future = _executor.submit(_run)
    return result
