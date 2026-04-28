"""brain/agents/code_agent.py

CodeAgent: write → run → verify → fix loop.
Uses MODEL_HEAVY (qwen2.5:32b) — tasks require full reasoning.
Max MAX_ITER LLM round-trips per task.

При route='code' сначала проверяет: есть ли в запросе готовый код или путь к .py.
Если нет — возвращает запрос прислать код/путь, не галлюцинирует.
"""
from __future__ import annotations

import json
import logging
import re

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

# Ключевые фразы, указывающие что пользователь хочет ЗАПУСТИТЬ существующий скрипт
_RUN_INTENT_PATTERNS = [
    r"запуст[иьи]",
    r"выполн[иьи]",
    r"run\s+this",
    r"run\s+script",
    r"execute",
    r"прогон[иь]",
]


def _extract_code_block(text: str) -> str | None:
    """Извлечь код из ```python ... ``` блока."""
    m = re.search(r"```(?:python)?\s*\n([\s\S]+?)\n```", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_py_path(text: str) -> str | None:
    """Извлечь путь к .py файлу из текста."""
    m = re.search(r"([A-Za-z]:[\\/][^\s'\"]+\.py|[\w./\\-]+\.py)", text)
    return m.group(1) if m else None


def _has_run_intent(text: str) -> bool:
    """Проверить, хочет ли пользователь запустить (а не написать) скрипт."""
    lower = text.lower()
    return any(re.search(p, lower) for p in _RUN_INTENT_PATTERNS)


def _strip_json_fences(raw: str) -> str:
    """FIX BUG-14: use regex to strip code fences, not str.strip() which strips chars."""
    text = raw.strip()
    text = re.sub(r'^```[\w]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return text.strip()


def run(
    task: str,
    history: list[dict] | None = None,
) -> str:
    """Execute a coding task iteratively. Returns final summary string."""

    # FIX: guard against hallucination when user asks to RUN a script but provides none
    if _has_run_intent(task):
        code_block = _extract_code_block(task)
        py_path = _extract_py_path(task)

        if code_block:
            # Код прямо в запросе — запустить через run_python
            logger.info("[code_agent] Run intent + inline code block detected, executing directly")
            result = run_python(code_block)
            ok = result.get("ok", False)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if ok:
                return f"✅ Скрипт выполнен.\n\nВывод:\n{stdout}" if stdout else "✅ Скрипт выполнен без вывода."
            return f"❌ Ошибка выполнения:\n{stderr or stdout}"

        if py_path:
            # Путь к файлу — читаем и запускаем
            logger.info("[code_agent] Run intent + .py path detected: %s", py_path)
            try:
                code = read_file(py_path)
            except Exception as e:
                return f"❌ Не удалось прочитать файл `{py_path}`: {e}"
            result = run_python(code)
            ok = result.get("ok", False)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if ok:
                return f"✅ `{py_path}` выполнен.\n\nВывод:\n{stdout}" if stdout else f"✅ `{py_path}` выполнен без вывода."
            return f"❌ Ошибка в `{py_path}`:\n{stderr or stdout}"

        # Намерение запустить, но ни кода ни пути нет — не галлюцинировать!
        logger.warning("[code_agent] Run intent but no code/path found in: %s", task[:120])
        return (
            "Пришли сам скрипт или путь к файлу .py.\n"
            "Например:\n"
            "  • вставь код в блок ```python ... ```\n"
            "  • или укажи путь: C:/jarvis/script.py"
        )

    # Нет намерения запустить — обычный агент написания кода
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    if history:
        for msg in history[-6:]:
            messages.append(msg)

    messages.append({"role": "user", "content": task})

    for iteration in range(MAX_ITER):
        logger.info("[code_agent] iteration %d/%d", iteration + 1, MAX_ITER)
        raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.1, "num_ctx": 16384})

        clean = _strip_json_fences(raw)

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
