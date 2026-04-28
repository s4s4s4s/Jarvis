"""brain/agents/code_agent.py

CodeAgent: write → run → verify → fix loop.
Uses MODEL_HEAVY (qwen2.5:32b) — tasks require full reasoning.
Max MAX_ITER LLM round-trips per task.
"""
from __future__ import annotations

import json
import logging

from brain.client import chat, MODEL_HEAVY
from tools.executor import run_python, run_pytest
from tools.file_ops import write_file, read_file

logger = logging.getLogger(__name__)

MAX_ITER = 5

SYSTEM = """Ты — автономный инженер-разработчик.
Тебе дана задача написать Python-код.

На каждом шаге верни ТОЛЬКО один JSON-объект (без markdown, без пояснений).

Доступные actions:
  {"action": "write",  "path": "<str>",  "content": "<str>"}
  {"action": "run",    "code": "<str>"}
  {"action": "test",   "path": "<str>"}     # запустить pytest
  {"action": "done",   "result": "<str>"}   # задача выполнена

Правила:
- Всегда проверяй код через run или test перед done.
- Если тест упал — исправь и повтори, не сдавайся раньше времени.
- Используй aiogram 3.x если нужен Telegram-бот (Router, не Dispatcher.register).
- Токены/секреты — только через os.environ, никогда хардкодом.
- При write сохраняй файл относительно C:/jarvis.
"""


def run(task: str, history: list[dict] | None = None) -> str:
    """Execute a coding task iteratively. Returns final summary string."""
    messages: list[dict] = [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": task},
    ]

    for iteration in range(MAX_ITER):
        logger.info("[code_agent] iteration %d/%d", iteration + 1, MAX_ITER)
        raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.1, "num_ctx": 16384})

        # strip possible markdown code fences
        clean = raw.strip()
        for fence in ("```json", "```"):
            clean = clean.strip(fence).strip()

        try:
            action_data = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[code_agent] invalid JSON: %s", raw[:200])
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": "Верни ТОЛЬКО валидный JSON-объект, без текста вокруг."})
            continue

        action = action_data.get("action")
        messages.append({"role": "assistant", "content": raw})
        logger.info("[code_agent] action=%s", action)

        if action == "done":
            return action_data.get("result", "Задача выполнена.")

        elif action == "write":
            result = write_file(action_data["path"], action_data["content"])
            messages.append({"role": "user", "content": f"Файл записан: {json.dumps(result, ensure_ascii=False)}"})

        elif action == "run":
            result = run_python(action_data["code"])
            messages.append({"role": "user", "content": f"Результат выполнения: {json.dumps(result, ensure_ascii=False)}"})

        elif action == "test":
            result = run_pytest(action_data.get("path", "."))
            messages.append({"role": "user", "content": f"Тесты: {json.dumps(result, ensure_ascii=False)}"})

        else:
            messages.append({"role": "user", "content": f"Неизвестный action '{action}'. Допустимые: write / run / test / done."})

    return "Превышен лимит итераций. Задача не завершена — попробуй разбить на подзадачи."
