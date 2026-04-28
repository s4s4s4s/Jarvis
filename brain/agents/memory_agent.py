from core.config import MAX_HISTORY
from brain.client import chat, MODEL_FAST
from brain.prompts import MEMORY_SYSTEM
from tools.memory import get_memory_context, recall_events # FIX BUG-5: Memory Route Fails to Recall Past Events


def run(query: str, history: list[dict]) -> str:
    context = get_memory_context(max_facts=30) or "(фактов пока нет)"
    recalled_events = recall_events(query) # FIX BUG-5: Memory Route Fails to Recall Past Events
    msgs = [
        {"role": "system", "content": MEMORY_SYSTEM},
        {"role": "system", "content": context},
    ]
    if recalled_events:
        msgs.append({"role": "assistant", "content": recalled_events}) # FIX BUG-5: Memory Route Fails to Recall Past Events
    msgs.extend(history[-(MAX_HISTORY * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 8192})