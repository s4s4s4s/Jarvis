# C:\jarvis\brain\agents\web_agent.py
from brain.client import chat, MODEL_FAST
from brain.prompts import WEB_SYSTEM
from tools import web_search as ws


def run(query: str, history: list[dict]) -> str:
    try:
        try:
            snippets = ws.search(query, history=history)
        except TypeError:
            snippets = ws.search(query)
    except Exception as e:
        return f"Сэр, поиск не удался: {e}"

    msgs = [
        {"role": "system", "content": WEB_SYSTEM},
        {"role": "system", "content": f"Результаты поиска:\n{snippets}"},
        {"role": "user", "content": query},
    ]
    return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 8192})
# === end of file: brain/agents/web_agent.py ===