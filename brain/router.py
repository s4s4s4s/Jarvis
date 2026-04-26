import re
import json
import time
from dataclasses import dataclass
from core.paths import ROUTER_LOG
from brain.client import chat, MODEL_ROUTER
from brain.prompts import ROUTER_SYSTEM


@dataclass
class RouterDecision:
    route: str
    confidence: float
    rewritten_query: str
    answer: str
    filler: str
    reason: str


# Регекспресс: фразы, которые без LLM идут в chat
_CHAT_SHORTCUTS = re.compile(
    r"^"
    r"(да||нет||хорошо||окей|ок|понял|поняла|поняли|понятно|спасибо|благодарю|спас|точно|верно|ясно|ладно|отлично|угу|не надо|согласен|согласна|не важно|всё верно|все верно|так и есть|так и было|ты прав|ты права|я тебя понял|я это знаю|жди|подожди|секунду|ничего себя|ничего себя)"
    r"[!.,?\s]*$",
    re.IGNORECASE,
)

# Регекспресс: время/дата → chat (всегда, не web)
_TIME_DATE_RE = re.compile(
    r"(который.{0,10}час|сколько.{0,10}время|сейчас.{0,5}час|часы|время сейчас|какое сейчас|какое время|текущее время|какая дата|сегодня|какое число|какой день|день недели|какой год)",
    re.IGNORECASE,
)


def _safe_parse(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        return json.loads(raw[start : end + 1])
    except Exception:
        return {}


def route(user_text: str, history: list[dict]) -> RouterDecision:
    t = user_text.strip()

    # 1. Regex-shortcut: короткие подтверждения/благодарности — сразу в chat
    if _CHAT_SHORTCUTS.match(t):
        print(f"[router] shortcut → chat: {t!r}")
        return RouterDecision(
            route="chat",
            confidence=1.0,
            rewritten_query=t,
            answer="Понял, Сэр.",
            filler="",
            reason="regex shortcut",
        )

    # 2. Regex: время/дата — chat, не web
    if _TIME_DATE_RE.search(t) and len(t) < 60:
        print(f"[router] time/date shortcut → chat: {t!r}")
        msgs = [{"role": "system", "content": ROUTER_SYSTEM}]
        # для времени/даты передаём 1 ход истории
        if history:
            msgs.append(history[-1])
        msgs.append({"role": "user", "content": t})
        raw = chat(MODEL_ROUTER, msgs, options={"temperature": 0.1, "num_ctx": 4096})
        data = _safe_parse(raw)
        # Форсируем chat если LLM вдруг решил иначе
        data["route"] = "chat"
        return RouterDecision(
            route="chat",
            confidence=float(data.get("confidence", 1.0) or 1.0),
            rewritten_query=data.get("rewritten_query", t),
            answer=data.get("answer", "") or "",
            filler=data.get("filler", "") or "",
            reason="time/date forced chat",
        )

    # 3. Основной LLM-роутер: только 1 последний ход истории
    msgs = [{"role": "system", "content": ROUTER_SYSTEM}]
    if history:
        msgs.append(history[-1])
    msgs.append({"role": "user", "content": t})

    raw = chat(MODEL_ROUTER, msgs, options={"temperature": 0.1, "num_ctx": 4096})
    data = _safe_parse(raw)

    decision = RouterDecision(
        route=data.get("route", "chat"),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rewritten_query=data.get("rewritten_query", t),
        answer=data.get("answer", "") or "",
        filler=data.get("filler", "") or "",
        reason=data.get("reason", "") or "",
    )

    try:
        with open(ROUTER_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": time.time(), "q": t, "raw": raw, "decision": decision.__dict__},
                ensure_ascii=False,
            ) + "\n")
    except Exception:
        pass

    return decision
