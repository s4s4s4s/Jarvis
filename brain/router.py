# C:\jarvis\brain\router.py
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


def _safe_parse(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        return json.loads(raw[start : end + 1])
    except Exception:
        return {}


def route(user_text: str, history: list[dict]) -> RouterDecision:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM}]
    msgs.extend(history[-8:])
    msgs.append({"role": "user", "content": user_text})

    raw = chat(MODEL_ROUTER, msgs, options={"temperature": 0.1, "num_ctx": 8192})
    data = _safe_parse(raw)

    decision = RouterDecision(
        route=data.get("route", "chat"),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rewritten_query=data.get("rewritten_query", user_text),
        answer=data.get("answer", "") or "",
        filler=data.get("filler", "") or "",
        reason=data.get("reason", "") or "",
    )

    try:
        with open(ROUTER_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": time.time(), "q": user_text, "raw": raw, "decision": decision.__dict__},
                ensure_ascii=False,
            ) + "\n")
    except Exception:
        pass

    return decision
# === end of file: brain/router.py ===