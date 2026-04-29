# brain/agents/planner.py
"""
PlannerAgent — агент многошаговых задач Jarvis.

Описание:
  Получает сложный запрос → LLM декомпозирует на шаги (JSON)
  → каждый шаг → существующий агент/инструмент
  → LLM синтезирует все результаты → ответ

Маршрут: route="plan"
Триггеры: "сделай X и потом Y", "найди X, запиши в файл Y", многошаговые задачи
"""
from __future__ import annotations

import json
import logging
from typing import Any

from brain.client import chat, MODEL_FAST, MODEL_HEAVY
from tools.registry import call_tool, list_tools

logger = logging.getLogger(__name__)

MAX_STEPS    = 8
STEP_TIMEOUT = 30.0

_PLAN_SYSTEM = """Ты — Jarvis, планировщик. Дана задача пользователя.
Доступные инструменты: {tools}

Разбей задачу на минимальное количество шагов (max {max_steps}).
Ответь ТОЛЬКО JSON (без пояснений), схема:
{{
  "plan": [
    {{"step": 1, "type": "tool",  "tool": "file.read",  "args": {{"path": "~/doc.txt"}}, "description": "что делаем"}},
    {{"step": 2, "type": "tool",  "tool": "file.write", "args": {{"path": "~/out.txt", "content": "{{step1_result}}"}}, "description": "запись"}},
    {{"step": 3, "type": "answer", "description": "синтез"}}
  ]
}}

Правила:
- type="tool" — вызов инструмента из списка
- type="answer" — финальный шаг, всегда последний
- {{stepN_result}} — подстановка результата шага N
- Не выдумывай инструменты — только из списка
"""

_SYNTH_SYSTEM = """Ты — Jarvis. Пользователь дал задачу.
План выполнен, у тебя есть результаты всех шагов.
Дай краткий естественный ответ на русском, как если рассказываешь что сделал."""


def _make_plan(query: str) -> list[dict] | None:
    tools_str = ", ".join(list_tools())
    system = _PLAN_SYSTEM.format(tools=tools_str, max_steps=MAX_STEPS)
    msgs = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": query},
    ]
    raw = chat(MODEL_HEAVY, msgs, options={"temperature": 0.1, "num_ctx": 8192})
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    try:
        data = json.loads(raw)
        return data.get("plan", [])
    except Exception as e:
        logger.error(f"[planner] plan parse error: {e} | raw={raw[:200]}")
        return None


def _substitute_results(args: dict, step_results: dict[int, Any]) -> dict:
    """Заменяет подстановки {stepN_result} в args значениями из предыдущих шагов."""
    result = {}
    for k, v in args.items():
        if isinstance(v, str):
            for step_num, step_res in step_results.items():
                placeholder = f"{{step{step_num}_result}}"
                if placeholder in v:
                    v = v.replace(placeholder, str(step_res)[:2000])
        result[k] = v
    return result


def run(query: str, history: list[dict]) -> str:
    """
    Вход: запрос пользователя.
    Выход: строка — ответ для голосового ассистента.
    """
    logger.info(f"[planner] Новая задача: {query[:80]}")

    plan = _make_plan(query)
    if not plan:
        return "Сэр, не удалось составить план выполнения."

    logger.info(f"[planner] План составлен: {len(plan)} шагов")
    step_results: dict[int, Any] = {}
    errors: list[str] = []

    for step in plan:
        step_num  = step.get("step", 0)
        step_type = step.get("type", "")
        desc      = step.get("description", "")

        if step_type == "answer":
            # Финальный синтез
            break

        if step_type == "tool":
            tool_name = step.get("tool", "")
            raw_args  = step.get("args", {})
            args      = _substitute_results(raw_args, step_results)

            logger.info(f"[planner] Шаг {step_num}: {tool_name}({args}) — {desc}")
            result = call_tool(tool_name, args)

            if result.ok:
                step_results[step_num] = result.data
            else:
                err = f"Шаг {step_num} ({tool_name}): {result.error}"
                logger.warning(f"[planner] {err}")
                errors.append(err)
                step_results[step_num] = f"ERROR: {result.error}"
        else:
            logger.warning(f"[planner] Неизвестный тип шага: {step_type}")

    # Синтез
    context_parts = [f"Задача: {query}\n"]
    for step in plan:
        sn = step.get("step", 0)
        if step.get("type") == "tool" and sn in step_results:
            res = step_results[sn]
            res_str = json.dumps(res, ensure_ascii=False)[:1500] if not isinstance(res, str) else res[:1500]
            context_parts.append(f"Шаг {sn} [{step.get('tool')}]: {res_str}")
    if errors:
        context_parts.append(f"\nОшибки: {'; '.join(errors)}")

    msgs = [
        {"role": "system",  "content": _SYNTH_SYSTEM},
        {"role": "user",    "content": "\n".join(context_parts)},
    ]
    try:
        answer = chat(MODEL_FAST, msgs, options={"temperature": 0.3, "num_ctx": 8192})
    except Exception as e:
        answer = f"Сэр, план выполнен, но синтез не удался: {e}"

    logger.info(f"[planner] Готово.")
    return answer
