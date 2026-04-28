from __future__ import annotations

import atexit
import json
import logging
import re
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

logger = logging.getLogger(__name__)

_FALLBACK_TIMEOUT_MSG = "Сэр, не удалось получить ответ вовремя."
_FALLBACK_ERROR = "Сэр, инструмент вернул ошибку: {error}"

# Tools whose full output should never be read aloud verbatim
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
}

# Max chars before we trim for TTS (no LLM call)
_VOICE_MAX_CHARS = 220


@dataclass
class AskResult:
    filler: str = ""
    _text:        str = field(default="", repr=False)
    _future:      Future | None = field(default=None, repr=False)
    _answer:      str = field(default="", repr=False)   # full answer → shown in chat
    _voice_reply: str = field(default="", repr=False)   # short phrase → spoken aloud

    def get_answer(self, timeout: float = 120.0) -> str:
        """Full answer for the chat UI."""
        if self._future is not None:
            try:
                self._answer, self._voice_reply = self._future.result(timeout=timeout)
            except Exception as e:
                logger.error("[ask] Future error: %s", e)
                self._answer = _FALLBACK_TIMEOUT_MSG
                self._voice_reply = _FALLBACK_TIMEOUT_MSG
            self._future = None
        return self._answer

    def get_voice_reply(self, timeout: float = 120.0) -> str:
        """Short phrase for TTS. Auto-populated after get_answer()."""
        if self._future is not None:
            self.get_answer(timeout=timeout)
        return self._voice_reply or self._answer


def _route(text: str) -> dict[str, Any]:
    msgs = [
        {"role": "system", "content": ROUTER_SYSTEM},
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
    """
    Rule-based TTS trimmer. Zero extra LLM calls.

    Strategy (in order):
    1. Strip markdown (headers, bold, bullets, code fences).
    2. For known long-output tools: keep only first 1-2 sentences.
    3. For any answer > _VOICE_MAX_CHARS: keep first 1-2 sentences.
    4. If still too long after 2 sentences: hard-cut at _VOICE_MAX_CHARS + ellipsis.
    """
    # 1. Strip markdown
    text = answer
    text = re.sub(r"```[\s\S]*?```", "", text)          # code blocks
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)  # headers
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text) # bold/italic
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.M)   # bullets
    text = re.sub(r"\|[^\n]+", "", text)                 # table rows
    text = re.sub(r"\n{2,}", " ", text)                  # collapse newlines
    text = re.sub(r"\s{2,}", " ", text).strip()

    if not text:
        # Fallback: strip everything, take raw first 120 chars
        text = re.sub(r"[`*#|\n]", " ", answer).strip()
        return text[:120].strip()

    # 2-3. For tool results OR long answers: take 1-2 sentences
    if tool_name in _VOICE_TRUNCATE_TOOLS or len(text) > _VOICE_MAX_CHARS:
        # Split on sentence-ending punctuation followed by space/end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for i, sent in enumerate(sentences):
            candidate = (result + " " + sent).strip() if result else sent
            if i >= 2 or len(candidate) > _VOICE_MAX_CHARS:
                if not result:
                    # Even 1st sentence is too long — hard cut
                    result = candidate[:_VOICE_MAX_CHARS].rstrip() + "..."
                break
            result = candidate
        text = result or text[:_VOICE_MAX_CHARS].rstrip() + "..."

    # 4. Final hard cap
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

    from brain.agents.chat import run as chat_run
    return chat_run(text, history)


def ask_llm(text: str) -> AskResult:
    history = hist.snapshot()
    hist.append("user", text)

    t_route0 = time.monotonic()
    try:
        route_data = _route(text)
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

    result = AskResult(filler=route_data.get("filler", ""), _text=text)
    tool_name = route_data.get("tool")

    def _run() -> tuple[str, str]:
        t0 = time.monotonic()
        answer = _dispatch(route_data, text, history)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        hist.append("assistant", answer)
        log_route(
            text=text,
            route=route_data["route"],
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

        # Build voice reply — rule-based, zero extra LLM calls
        voice_reply = _trim_for_voice(answer, tool_name)
        logger.debug("[ask] voice_reply (%d chars): %s", len(voice_reply), voice_reply[:80])

        return answer, voice_reply

    result._future = _executor.submit(_run)
    return result
