from core.config import MAX_HISTORY
from brain.client import chat, MODEL_FAST
from brain.prompts import WEB_SYSTEM
from tools.memory import get_memory_context
from tools.web_search import web_search


def run(query: str, history: list[dict]) -> str:
    try:
        snippets = web_search(query)
    except Exception as e:
        # FIX (аудит 6): раньше был хардкод "Сэр, поиск не удался: ...".
        # Теперь LLM формулирует ответ об ошибке. Хардкод — только fallback.
        try:
            err_msgs = [
                {"role": "system", "content": WEB_SYSTEM},
                {"role": "user", "content": (
                    f"Запрос пользователя: {query}\n\n"
                    f"Веб-поиск завершился с ошибкой:\n{e}\n\n"
                    f"Сообщи пользователю, что не удалось найти информацию в интернете, кратко."
                )},
            ]
            return chat(MODEL_FAST, err_msgs, options={"temperature": 0.3, "num_ctx": 4096})
        except Exception:
            return f"Сэр, поиск не удался: {e}"

    msgs = [
        {"role": "system", "content": WEB_SYSTEM},
        {"role": "system", "content": f"Результаты поиска:\n{snippets}"},
    ]

    # Память пользователя — для персонализации веб-ответа
    mem = get_memory_context(max_facts=10)
    if mem:
        msgs.append({"role": "system", "content": mem})

    # Полная история — чтобы уточняющие вопросы ('a теперь про это подробнее') работали
    msgs.extend(history[-(MAX_HISTORY * 2):])
    msgs.append({"role": "user", "content": query})
    return chat(MODEL_FAST, msgs, options={"temperature": 0.2, "num_ctx": 8192})
