# C:\jarvis\brain\agents\memory_agent.py
from brain.client import chat, MODEL_FAST
from brain.prompts import MEMORY_SYSTEM
from tools.memory import load_facts


def run(query: str, history: list[dict]) -> str:
    facts = load_facts()
    facts_block = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "(фактов нет)"
    msgs = [
        {"role": "system", "content": MEMORY_SYSTEM},
        {"role": "system", "content": f"Известные факты о Сэре:\n{facts_block}"},
    ]
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 8192})
# === end of file: brain/agents/memory_agent.py ===