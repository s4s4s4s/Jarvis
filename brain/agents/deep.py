# C:\jarvis\brain\agents\deep.py
from brain.client import chat, MODEL_HEAVY
from brain.prompts import DEEP_SYSTEM


def run(query: str, history: list[dict]) -> str:
    msgs = [{"role": "system", "content": DEEP_SYSTEM}]
    msgs.extend(history[-10:])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_HEAVY, msgs, options={"temperature": 0.4, "num_ctx": 16384})
# === end of file: brain/agents/deep.py ===