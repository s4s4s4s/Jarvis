from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from brain.client import chat, MODEL_ROUTER, MODEL_FAST
from brain.prompts import ROUTER_SYSTEM, TOOL_FORMAT_SYSTEM
from brain import history as hist
from brain.logger import log_route
from brain.router_embed import route_embed, eager_load

_executor         = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")
_project_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jarvis-project")
atexit.register(lambda: _executor.shutdown(wait=False, cancel_futures=True))
atexit.register(lambda: _project_executor.shutdown(wait=False, cancel_futures=True))

_hist_lock = threading.Lock()

try:
    threading.Thread(target=eager_load, daemon=True, name="jarvis-embed-eager-load").start()
except Exception:
    pass

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT   = "get_answer"
_FALLBACK_ERROR = "Сэр, инструмент вернул ошибку: {error}"
_NO_MEMORY_ROUTES = frozenset({"tool", "web", "feedback"})


@dataclass
class AskResult:
    filler: str = ""
    text: str = ""
    _future: Future | None = field(default=None, repr=False)
    _answer: str = field(default="", repr=False)

    def get_answer(self, timeout: float = 120.0) -> str:
        if self._future is not None:
            try:
                self._answer = self._future.result(timeout=timeout)
            except Exception as e:
                self._answer = _format_tool_error(
                    self.text or "запрос",
                    _TOOL_TIMEOUT,
                    str(e),
                )
            self._future = None
        return self._answer


def _route_llm(text: str) -> dict[str, Any]:
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
        "route":      data.get("route", "chat"),
        "tool":       data.get("tool"),
        "tool_args":  data.get("tool_args") or {},
        "confidence": data.get("confidence", 0.0),
        "filler":     data.get("filler") or "",
        "reason":     data.get("reason") or "",
        "_source":    "llm",
    }


def _route_smart(text: str) -> dict[str, Any]:
    result = route_embed(text)
    if result is not None:
        logger.debug(f"[router] embed hit: route={result['route']} conf={result['confidence']}")
        return result
    logger.debug("[router] embed uncertain → LLM fallback")
    return _route_llm(text)


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
        logger.error(f"LLM error in _format_tool_error for tool '{tool_name}': {e}")
        return _FALLBACK_ERROR.format(error=error)


def _dispatch(route_data: dict[str, Any], text: str, history: list[dict]) -> tuple[str, bool]:
    route = route_data["route"]

    if route == "feedback":
        from brain.explicit_feedback import user_says_correct, user_says_wrong
        tool = route_data.get("tool") or ""
        if tool == "feedback.correct":
            return user_says_correct(), True
        return user_says_wrong(), True

    if route == "plan":
        from brain.agents.planner import run as planner_run
        return planner_run(text, history), True

    if route == "extend":
        from brain.agents.self_extend import run as extend_run
        return extend_run(text, history), True

    if route == "project":
        from brain.agents.project import run as project_run
        return project_run(text, history), True

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
            return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 4096}), True
        return _format_tool_error(text, route_data.get("tool"), result.error or "неизвестная ошибка"), False

    if route == "web":
        from brain.agents.web_agent import run as web_run
        return web_run(text, history), True

    if route == "deep":
        from brain.agents.deep import run as deep_run
        return deep_run(text, history), True

    if route == "memory":
        from brain.agents.memory_agent import run as memory_run
        return memory_run(text, history), True

    from brain.agents.chat import run as chat_run
    return chat_run(text, history), True


def ask_llm(text: str) -> AskResult:
    # 1) pending clarify: сначала уточнения проекта
    try:
        from brain.agents.project_clarify import has_pending_clarify, looks_like_clarify_answer
        from brain.agents.project_clarify import provide_clarify_answers as _provide_clarify_answers
        if has_pending_clarify() and looks_like_clarify_answer(text):
            clarify_result = AskResult(filler="Уточняю проект...", text=text)
            clarify_result._future = _project_executor.submit(_provide_clarify_answers, text)
            return clarify_result
    except Exception as exc:
        logger.warning(f"[ask] clarify pre-check failed: {exc}")

    # 2) pending credentials: после уточнений и только если проект уже их запросил
    try:
        from brain.agents.project_creds import has_pending_creds, looks_like_creds
        from brain.agents.project_creds import provide_credentials as _provide_creds
        if has_pending_creds() and looks_like_creds(text):
            creds_result = AskResult(filler="Сохраняю данные...", text=text)
            creds_result._future = _executor.submit(_provide_creds, text)
            return creds_result
    except Exception as exc:
        logger.warning(f"[ask] credentials pre-check failed: {exc}")

    history = hist.snapshot()

    t_route0 = time.monotonic()
    try:
        route_data = _route_smart(text)
    except Exception as e:
        route_data = {
            "route":      "chat",
            "tool":       None,
            "tool_args":  {},
            "confidence": 0.0,
            "filler":     "",
            "reason":     f"router error: {e}",
            "_source":    "error",
        }
    route_ms = int((time.monotonic() - t_route0) * 1000)

    result = AskResult(filler=route_data.get("filler", ""), text=text)

    def _run() -> str:
        t0 = time.monotonic()
        answer, tool_ok = _dispatch(route_data, text, history)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        with _hist_lock:
            hist.append("user", text)
            hist.append("assistant", answer)

        try:
            from brain.implicit_eval import evaluate as implicit_eval
            from brain.feedback_store import record as fb_record
            from brain.explicit_feedback import set_last_id

            outcome, reason = implicit_eval(
                route=route_data["route"],
                tool=route_data.get("tool"),
                confidence=route_data.get("confidence", 0.0),
                answer=answer,
                tool_ok=tool_ok,
            )
            record_id = fb_record(
                text=text,
                route=route_data["route"],
                tool=route_data.get("tool"),
                confidence=route_data.get("confidence", 0.0),
                source=route_data.get("_source", "llm"),
                answer=answer,
                outcome=outcome,
                reason=reason,
            )
            set_last_id(record_id)
        except Exception as e:
            logger.error(f"[ask] feedback error: {e}")

        log_route(
            text=text,
            route=route_data["route"],
            tool=route_data.get("tool"),
            confidence=route_data.get("confidence", 0.0),
            reason=route_data.get("reason", ""),
            answer_ms=elapsed_ms + route_ms,
        )
        if route_data.get("route") not in _NO_MEMORY_ROUTES:
            try:
                from tools.memory import extract_and_save_async
                extract_and_save_async(text, answer)
            except Exception as e:
                logger.error(f"Memory extraction failed: {e}")
        return answer

    executor = _project_executor if route_data.get("route") == "project" else _executor
    result._future = executor.submit(_run)
    return result
