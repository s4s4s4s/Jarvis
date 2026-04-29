from __future__ import annotations

from core.config import MAX_HISTORY
from brain.client import chat, MODEL_HEAVY
from brain.prompts import build_deep_system
from tools.memory import get_memory_context


def run(query: str, history: list[dict]) -> str:
    # FIX: use build_deep_system() for fresh date injection on every call
    msgs = [{"role": "system", "content": build_deep_system()}]

    # Инжекция долгосрочной памяти — для персонализации глубоких ответов
    mem = get_memory_context(max_facts=15)
    if mem:
        msgs.append({"role": "system", "content": mem})

    msgs.extend(history[-(MAX_HISTORY * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_HEAVY, msgs, options={"temperature": 0.4, "num_ctx": 16384})
