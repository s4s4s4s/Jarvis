from core.config import MAX_HISTORY
from brain.client import chat, MODEL_FAST
from brain.prompts import WEB_SYSTEM
from tools.web_search import web_search


def run(query: str, history: list[dict]) -> str:
    try:
        snippets = web_search(query)
    except Exception as e:
        return f"Сэр, поиск не удался: {e}"

    msgs = [
        {"role": "system", "content": WEB_SYSTEM},
        {"role": "system", "content": f"Результаты поиска:\n{snippets}"},
    ]
    msgs.extend(history[-(min(MAX_HISTORY, 2) * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 8192})
