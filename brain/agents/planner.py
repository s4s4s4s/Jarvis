"""
PlannerAgent — агент многошаговых задач Jarvis.

Hardening v2:
  - Валидация JSON-схемы плана (jsonschema)
  - Retry x2 с упрощённым prompt если plan не парсится
  - Защита от кривых подстановок {stepN_result}
  - Timeout на каждый шаг (STEP_TIMEOUT)
  - Никогда не падает молча — всегда возвращает строку

fix #6: _validate_plan() принимает known_tools как параметр (list | None),
а list_tools() вызывается один раз в _make_plan() с try/except вокруг.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from brain.client import chat, MODEL_FAST, MODEL_HEAVY
from tools.registry import call_tool, list_tools

logger = logging.getLogger(__name__)

MAX_STEPS     = 8
STEP_TIMEOUT  = 30.0   # секунд на один шаг
PLAN_RETRIES  = 2       # попыток получить валидный план

# ── JSON-схема плана ─────────────────────────────────────────────────────────────────────────
_PLAN_SCHEMA = {
    "type": "object",
    "required": ["plan"],
    "properties": {
        "plan": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["step", "type"],
                "properties": {
                    "step":        {"type": "integer", "minimum": 1},
                    "type":        {"type": "string", "enum": ["tool", "answer"]},
                    "tool":        {"type": "string"},
                    "args":        {"type": "object"},
                    "description": {"type": "string"},
                },
            },
        }
    },
}

# ── промпты ───────────────────────────────────────────────────────────────────────────────────────
_PLAN_SYSTEM = """Ты — Jarvis, планировщик. Дана задача пользователя.
Доступные инструменты: {tools}

Разбей задачу на минимальное количество шагов (max {max_steps}).
Ответь ТОЛЬКО JSON (без пояснений, без markdown-блоков), схема:
{{"plan": [
  {{"step": 1, "type": "tool",   "tool": "file.read",  "args": {{"path": "~/doc.txt"}}, "description": "читаем файл"}},
  {{"step": 2, "type": "tool",   "tool": "file.write", "args": {{"path": "~/out.txt", "content": "{{step1_result}}"}}, "description": "пишем"}},
  {{"step": 3, "type": "answer", "description": "синтез"}}
]}}

Правила:
- type="tool"   — вызов инструмента из списка выше
- type="answer" — финальный шаг, всегда последний, без "tool"
- {{stepN_result}} — вставляет результат шага N (только строка, в значении "args")
- НЕ выдумывай инструменты — только из списка
- Максимум {max_steps} шагов включая финальный
"""

_PLAN_RETRY_SYSTEM = """Ты — JSON-генератор. Верни ТОЛЬКО валидный JSON без пояснений.
Схема: {{"plan": [{{"step":1,"type":"tool","tool":"имя","args":{{}},"description":"..."}}, ..., {{"step":N,"type":"answer","description":"синтез"}}]}}
Доступные инструменты: {tools}
Задача: {query}
"""

_SYNTH_SYSTEM = """Ты — Jarvis. Пользователь дал задачу.
План выполнен, у тебя есть результаты всех шагов.
Дай краткий естественный ответ на русском, как если рассказываешь что сделал."""


# ── утилиты ──────────────────────────────────────────────────────────────────────────────────────────

def _strip_markdown(raw: str) -> str:
    """\u0423\u0431\u0438\u0440\u0430\u0435\u0442 ```json ... ``` \u043e\u0431\u0451\u0440\u0442\u043a\u0438."""
    raw = raw.strip()
    raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return raw.strip()


def _validate_plan(plan: list[dict], known_tools: list[str] | None = None) -> tuple[bool, str]:
    """П\u0440\u043e\u0432\u0435\u0440\u044f\u0435\u0442 \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0443 \u043f\u043b\u0430\u043d\u0430.

    fix #6: known_tools \u043f\u0435\u0440\u0435\u0434\u0430\u0451\u0442\u0441\u044f \u0441\u043d\u0430\u0440\u0443\u0436\u0438 \u0447\u0442\u043e\u0431\u044b \u043d\u0435 \u0432\u044b\u0437\u044b\u0432\u0430\u0442\u044c list_tools() \u043d\u0430 \u043a\u0430\u0436\u0434\u043e\u043c \u0448\u0430\u0433\u0435.
    \u0415\u0441\u043b\u0438 None \u2014 \u0438\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442\u044b \u043d\u0435 \u043f\u0440\u043e\u0432\u0435\u0440\u044f\u044e\u0442\u0441\u044f (fallback \u0434\u043b\u044f \u043e\u0431\u0440\u0430\u0442\u043d\u043e\u0439 \u0441\u043e\u0432\u043c\u0435\u0441\u0442\u0438\u043c\u043e\u0441\u0442\u0438).
    """
    if not isinstance(plan, list) or len(plan) == 0:
        return False, "план пуст или не список"
    if len(plan) > MAX_STEPS:
        return False, f"слишком много шагов: {len(plan)} > {MAX_STEPS}"

    seen_steps: set[int] = set()
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return False, f"шаг {i} не является объектом"
        step_num = step.get("step")
        if not isinstance(step_num, int) or step_num < 1:
            return False, f"шаг {i}: поле 'step' должно быть int >= 1"
        if step_num in seen_steps:
            return False, f"дублирующийся номер шага: {step_num}"
        seen_steps.add(step_num)
        step_type = step.get("type")
        if step_type not in ("tool", "answer"):
            return False, f"шаг {step_num}: неизвестный type='{step_type}'"
        if step_type == "tool":
            tool_name = step.get("tool", "")
            if not tool_name:
                return False, f"шаг {step_num}: tool не указан"
            if known_tools is not None and tool_name not in known_tools:
                return False, f"шаг {step_num}: инструмент '{tool_name}' не существует. Доступные: {known_tools}"
            if not isinstance(step.get("args", {}), dict):
                return False, f"шаг {step_num}: args должен быть объектом"

    if plan[-1].get("type") != "answer":
        return False, "последний шаг должен быть type='answer'"

    return True, ""


def _safe_substitute(args: dict, step_results: dict[int, Any]) -> dict:
    """
    Безопасная подстановка {stepN_result}.
    - Только в строковых значениях
    - Обрезает результат до 2000 символов
    - Если шаг ещё не выполнен — оставляет placeholder как есть (не падает)
    """
    PLACEHOLDER_RE = re.compile(r'\{step(\d+)_result\}')
    result: dict = {}
    for k, v in args.items():
        if not isinstance(v, str):
            result[k] = v
            continue
        def _repl(m: re.Match) -> str:
            sn = int(m.group(1))
            if sn in step_results:
                val = step_results[sn]
                if not isinstance(val, str):
                    val = json.dumps(val, ensure_ascii=False)
                return val[:2000]
            logger.warning(f"[planner] substitute: шаг {sn} ещё не выполнен")
            return m.group(0)
        result[k] = PLACEHOLDER_RE.sub(_repl, v)
    return result


class _StepTimeoutError(Exception):
    pass


def _call_tool_with_timeout(tool_name: str, args: dict, timeout: float) -> Any:
    result_holder: list = [None]
    exc_holder:    list = [None]

    def _worker():
        try:
            result_holder[0] = call_tool(tool_name, args)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise _StepTimeoutError(f"инструмент '{tool_name}' не ответил за {timeout:.0f}с")
    if exc_holder[0]:
        raise exc_holder[0]
    return result_holder[0]


# ── построение плана (с retry) ──────────────────────────────────────────────────────────────────────

def _make_plan(query: str) -> tuple[list[dict] | None, str]:
    """
    Возвращает (план, "") при успехе или (None, причина_ошибки).
    Делает до PLAN_RETRIES попыток.

    fix #6: list_tools() вызывается один раз с try/except,
    результат передаётся в _validate_plan() аргументом.
    """
    # fix #6: один вызов list_tools() на всё планирование
    try:
        known_tools: list[str] | None = list_tools()
    except Exception as e:
        logger.warning(f"[planner] list_tools() failed: {e} — инструменты не будут валидированы")
        known_tools = None  # fallback: валидация типов остаётся, проверка tool_name отключается

    tools_str = ", ".join(known_tools) if known_tools else "недоступны"
    last_error = "неизвестная ошибка"

    for attempt in range(1, PLAN_RETRIES + 1):
        if attempt == 1:
            system = _PLAN_SYSTEM.format(tools=tools_str, max_steps=MAX_STEPS)
            msgs = [
                {"role": "system", "content": system},
                {"role": "user",   "content": query},
            ]
        else:
            system = _PLAN_RETRY_SYSTEM.format(tools=tools_str, query=query)
            msgs = [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Предыдущая попытка провалилась: {last_error}. Попробуй снова."},
            ]

        try:
            raw = chat(MODEL_HEAVY, msgs, options={"temperature": 0.1, "num_ctx": 8192})
            raw = _strip_markdown(raw)
            data = json.loads(raw)
            plan = data.get("plan", [])
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e} | raw={raw[:300]}"
            logger.warning(f"[planner] attempt {attempt}: {last_error}")
            continue
        except Exception as e:
            last_error = f"LLM error: {e}"
            logger.error(f"[planner] attempt {attempt}: {last_error}")
            continue

        valid, reason = _validate_plan(plan, known_tools=known_tools)
        if not valid:
            last_error = f"schema error: {reason}"
            logger.warning(f"[planner] attempt {attempt}: {last_error}")
            continue

        logger.info(f"[planner] план принят с попытки {attempt}: {len(plan)} шагов")
        return plan, ""

    return None, last_error


# ── основной run ──────────────────────────────────────────────────────────────────────────────────

def run(query: str, history: list[dict]) -> str:
    """
    Вход: запрос пользователя.
    Выход: строка — ответ для голосового ассистента.
    Никогда не бросает исключение.
    """
    logger.info(f"[planner] Новая задача: {query[:80]}")

    try:
        plan, err = _make_plan(query)
    except Exception as e:
        logger.exception("[planner] _make_plan crashed")
        return f"Сэр, планировщик упал: {e}"

    if plan is None:
        logger.error(f"[planner] план не составлен: {err}")
        return f"Сэр, не удалось составить план: {err}"

    logger.info(f"[planner] выполняю {len(plan)} шагов")
    step_results: dict[int, Any] = {}
    errors: list[str] = []

    for step in plan:
        step_num  = step.get("step", 0)
        step_type = step.get("type", "")
        desc      = step.get("description", "")

        if step_type == "answer":
            break

        if step_type == "tool":
            tool_name = step.get("tool", "")
            raw_args  = step.get("args") or {}

            if not isinstance(raw_args, dict):
                err_msg = f"Шаг {step_num}: args не dict, получили {type(raw_args).__name__}"
                logger.warning(f"[planner] {err_msg}")
                errors.append(err_msg)
                step_results[step_num] = f"ERROR: {err_msg}"
                continue

            args = _safe_substitute(raw_args, step_results)
            logger.info(f"[planner] Шаг {step_num}: {tool_name}({list(args.keys())}) — {desc}")

            try:
                result = _call_tool_with_timeout(tool_name, args, timeout=STEP_TIMEOUT)
            except _StepTimeoutError as e:
                err_msg = f"Шаг {step_num} ({tool_name}): таймаут — {e}"
                logger.warning(f"[planner] {err_msg}")
                errors.append(err_msg)
                step_results[step_num] = f"ERROR: timeout"
                continue
            except Exception as e:
                err_msg = f"Шаг {step_num} ({tool_name}): неожиданная ошибка — {e}"
                logger.exception(f"[planner] {err_msg}")
                errors.append(err_msg)
                step_results[step_num] = f"ERROR: {e}"
                continue

            if result.ok:
                step_results[step_num] = result.data
            else:
                err_msg = f"Шаг {step_num} ({tool_name}): {result.error}"
                logger.warning(f"[planner] {err_msg}")
                errors.append(err_msg)
                step_results[step_num] = f"ERROR: {result.error}"
        else:
            logger.warning(f"[planner] неизвестный тип шага: {step_type}")

    # ── синтез ──
    context_parts = [f"Задача: {query}\n"]
    for step in plan:
        sn = step.get("step", 0)
        if step.get("type") == "tool" and sn in step_results:
            res = step_results[sn]
            if not isinstance(res, str):
                res = json.dumps(res, ensure_ascii=False)[:1500]
            else:
                res = res[:1500]
            context_parts.append(f"Шаг {sn} [{step.get('tool')}]: {res}")
    if errors:
        context_parts.append(f"\nОшибки при выполнении: {'; '.join(errors)}")

    msgs = [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user",   "content": "\n".join(context_parts)},
    ]
    try:
        answer = chat(MODEL_FAST, msgs, options={"temperature": 0.3, "num_ctx": 8192})
    except Exception as e:
        logger.exception("[planner] синтез упал")
        answer = f"Сэр, план выполнен, но синтез не удался: {e}"

    logger.info("[planner] Готово.")
    return answer
