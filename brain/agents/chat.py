from core.config import MAX_HISTORY
from brain.client import chat, MODEL_FAST
from brain.prompts import CHAT_SYSTEM


def run(query: str, history: list[dict]) -> str:
    msgs = [{"role": "system", "content": CHAT_SYSTEM}]
    msgs.extend(history[-(MAX_HISTORY * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.3, "num_ctx": 8192})
