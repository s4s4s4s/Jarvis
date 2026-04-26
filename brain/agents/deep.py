from core.config import MAX_HISTORY
from brain.client import chat, MODEL_HEAVY
from brain.prompts import DEEP_SYSTEM


def run(query: str, history: list[dict]) -> str:
    msgs = [{"role": "system", "content": DEEP_SYSTEM}]
    msgs.extend(history[-(MAX_HISTORY * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_HEAVY, msgs, options={"temperature": 0.4, "num_ctx": 16384})
