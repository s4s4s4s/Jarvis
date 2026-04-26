# brain/agents/memory_agent.py
from brain.client import chat, MODEL_FAST
from brain.prompts import MEMORY_SYSTEM
from tools.memory import get_memory_context


def run(query: str, history: list[dict]) -> str:
    context = get_memory_context(max_facts=30) or "(фактов пока нет)"
    msgs = [
        {"role": "system", "content": MEMORY_SYSTEM},
        {"role": "system", "content": context},
    ]
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 8192})
# === end of file: brain/agents/memory_agent.py ===