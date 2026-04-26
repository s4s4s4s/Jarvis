# C:\jarvis\brain\ask.py
import threading
from dataclasses import dataclass, field
from typing import Callable

from brain import history as hist
from brain.router import route, RouterDecision
from brain.agents import chat as agent_chat
from brain.agents import memory_agent, web_agent, deep as agent_deep
from tools.memory import extract_and_save_async

CHAT_SHORTCUT_CONF = 0.85


@dataclass
class AskResult:
    filler: str
    route: str
    confidence: float
    _event: threading.Event = field(default_factory=threading.Event)
    _answer: str = ""

    def get_answer(self, timeout: float | None = None) -> str:
        self._event.wait(timeout)
        return self._answer


def _finalize(result: AskResult, user_text: str, answer: str) -> None:
    result._answer = answer
    with hist.lock():
        pass
    hist.append("user", user_text)
    hist.append("assistant", answer)
    try:
        extract_and_save_async(user_text, answer)
    except Exception:
        pass
    result._event.set()


def _run_agent(decision: RouterDecision, user_text: str, history: list[dict], result: AskResult) -> None:
    agents: dict[str, Callable[[str, list[dict]], str]] = {
        "chat": agent_chat.run,
        "memory": memory_agent.run,
        "web": web_agent.run,
        "deep": agent_deep.run,
    }
    fn = agents.get(decision.route, agent_chat.run)
    try:
        answer = fn(decision.rewritten_query or user_text, history)
    except Exception as e:
        answer = f"Сэр, произошла ошибка в агенте {decision.route}: {e}"
    _finalize(result, user_text, answer)


def ask_llm(user_text: str) -> AskResult:
    history = hist.snapshot()
    decision = route(user_text, history)

    # chat-shortcut
    if (
        decision.route == "chat"
        and decision.answer
        and decision.confidence >= CHAT_SHORTCUT_CONF
    ):
        result = AskResult(filler="", route="chat", confidence=decision.confidence)
        _finalize(result, user_text, decision.answer)
        return result

    result = AskResult(filler=decision.filler, route=decision.route, confidence=decision.confidence)
    t = threading.Thread(target=_run_agent, args=(decision, user_text, history, result), daemon=True)
    t.start()
    return result
# === end of file: brain/ask.py ===