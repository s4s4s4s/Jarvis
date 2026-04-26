from __future__ import annotations

from core.config import MAX_HISTORY
from brain.client import chat, MODEL_FAST
from brain.prompts import CHAT_SYSTEM
from tools.memory import get_memory_context


def run(query: str, history: list[dict]) -> str:
    msgs = [{"role": "system", "content": CHAT_SYSTEM}]

    # Inject long-term memory into every chat turn
    mem = get_memory_context(max_facts=15)
    if mem:
        msgs.append({"role": "system", "content": mem})

    msgs.extend(history[-(MAX_HISTORY * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.3, "num_ctx": 8192})
