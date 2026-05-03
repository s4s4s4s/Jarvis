"""
brain/agents/self_extend.py — SelfExtendAgent (Level 3)

Что делает:
  1. Пользователь описывает нужный инструмент естественным языком
  2. LLM генерирует имя + код инструмента (tools/<name>.py)
  3. Проходит security-скан: запрещены опасные паттерны (exec, subprocess, rm -rf и т.)
  4. Запускает AST-парсинг — код должен быть синтаксически валидным Python
  5. Сохраняет файл в tools/<name>.py
  6. Добавляет запись в tools/registry.py (секция _TOOL_MAP)
  7. Запускает быстрый smoke-тест через call_tool
  8. Логирует результат в logs/self_extend.jsonl

Маршрут: route="extend"
Триггер: "добавь инструмент X", "напиши инструмент для Y", "создай tool ..."
"""
from __future__ import annotations

import ast
import importlib
import json
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path

from brain.client import chat, MODEL_HEAVY
from core.paths import ROOT, LOGS_DIR
from tools.registry import call_tool, list_tools

logger = logging.getLogger(__name__)

EXTEND_LOG = LOGS_DIR / "self_extend.jsonl"
TOOLS_DIR  = ROOT / "tools"

# ─── запрещённые паттерны ─────────────────────────────────────────────────────────────────────
_DANGEROUS_PATTERNS = [
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"__import__",
    r"subprocess",
    r"os\.system",
    r"os\.popen",
    r"shutil\.rmtree",
    r"os\.remove.*\*",
    r"open\s*\([^)]*['\"]в['\"].*\.\.\.",   # запись в произвольные пути
    r"socket\.connect",
    r"requests\.post.*password",
    r"import\s+ctypes",
    r"winreg",
]


def _security_check(code: str) -> tuple[bool, str]:
    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, code, re.IGNORECASE):
            return False, f"Обнаружен опасный паттерн: {pat}"
    return True, ""


def _ast_check(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"


# ─── промпты ──────────────────────────────────────────────────────────────────────────────────────

_DESIGN_SYSTEM = """Ты — Jarvis, архитектор инструментов. Пользователь хочет добавить новый инструмент.
Уже есть: {existing_tools}

Ответь ONLY JSON без пояснений:
{{
  "tool_name": "<категория>.<действие>",
  "filename": "tools/<имя_файла>.py",
  "entry_fn": "<имя_функции>",
  "description": "<одна строка>",
  "args": {{"param": "type"}},
  "returns": "<что возвращает>"
}}

Правила:
- tool_name формат "category.action" (например: "screenshot.take", "audio.record")
- Имя не должно совпадать с уже существующими
- entry_fn: единственная функция, принимает kwargs, возвращает строку или dict
"""

_CODE_SYSTEM = """Ты — Jarvis, пишешь Python инструмент для локального Windows ассистента.

Напиши файл tools/{filename} с одной публичной функцией {entry_fn}(**kwargs).

Требования:
- Только stdlib + зависимости уже есть в проекте или описанные в задаче
- Функция возвращает str или dict, никогда не падает без try/except
- Не использовать: exec, eval, subprocess, os.system, shutil.rmtree
- Docstring на русском одной строкой
- Верни ТОЛЬКО код (без markdown, без пояснений)

Задача: {description}
entry_fn: {entry_fn}
"""

_REGISTRY_PATCH_SYSTEM = """Ты — Python-редактор. Добавь запись в tools/registry.py.

Текущий _TOOL_MAP (ключи): {tool_keys}

Dобавь:
  "{tool_name}": обёртка вокруг {entry_fn} из {filename}

Верни ONLY JSON:
{{
  "import_line": "from tools.<module> import <entry_fn>",
  "wrapper_code": "def _call_<snake>(**kwargs):\n    return <entry_fn>(**kwargs)",
  "map_entry": "    \"{tool_name}\": _call_<snake>,"
}}
"""


# ─── логирование ────────────────────────────────────────────────────────────────────────────────────────

def _log_result(tool_name: str, status: str, detail: str, query: str) -> None:
    EXTEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts":        datetime.now().isoformat(timespec="seconds"),
        "tool_name": tool_name,
        "status":    status,   # created | failed | smoke_fail | security_fail
        "detail":    detail,
        "query":     query[:200],
    }
    with open(EXTEND_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── патч registry.py ──────────────────────────────────────────────────────────────────────────────────

def _patch_registry(tool_name: str, filename: str, entry_fn: str) -> tuple[bool, str]:
    """
    Добавляет импорт + wrapper + запись в _TOOL_MAP.
    Патч напрямую в файл, без LLM — детерминированный формат.

    fix #3: порядок анкоров проверяется в порядке приоритета (realia реальных anchor-ов в файле).
    fix #7: заменен некорректный lstrip(«tools/») на removeprefix(«tools/»).
    """
    registry_path = ROOT / "tools" / "registry.py"
    try:
        src = registry_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Не удалось прочитать registry.py: {e}"

    # имя модуля: tools/my_tool.py → my_tool
    module = Path(filename).stem
    snake  = tool_name.replace(".", "_").replace("-", "_")
    import_line  = f"from tools.{module} import {entry_fn}"
    wrapper_name = f"_call_{snake}"
    wrapper_code = (
        f"\ndef {wrapper_name}(**kwargs):\n"
        f"    return {entry_fn}(**kwargs)\n"
    )
    map_line = f'    "{tool_name}": {wrapper_name},'

    # не дублируем
    if import_line in src:
        return True, "уже зарегистрирован"

    # вставляем import перед блоком "# ── memory"
    # fix #3: проверяем несколько анкоров в порядке реального файла
    import_anchors = [
        "# ── memory",       # если есть секция памяти
        "from tools.memory import",  # первая строка блока memory (fix #3)
        "@dataclass",        # fallback: вставляем до первого dataclass
    ]
    placed_import = False
    for anchor in import_anchors:
        if anchor in src:
            src = src.replace(anchor, f"{import_line}\n{anchor}", 1)
            placed_import = True
            break
    if not placed_import:
        # ультра-fallback: добавляем после последнего import-блока
        lines = src.splitlines(keepends=True)
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                last_import_idx = i
        lines.insert(last_import_idx + 1, import_line + "\n")
        src = "".join(lines)

    # вставляем wrapper перед _TOOL_MAP
    map_anchor = "\n_TOOL_MAP:"
    if map_anchor in src:
        src = src.replace(map_anchor, wrapper_code + map_anchor, 1)
    else:
        src = src.replace("_TOOL_MAP: dict", wrapper_code + "_TOOL_MAP: dict", 1)

    # вставляем запись в _TOOL_MAP перед закрывающей скобкой
    closing = "\n}\n\n\ndef list_tools"
    if closing in src:
        src = src.replace(closing, f"\n{map_line}{closing}", 1)
    else:
        src = re.sub(
            r'(\n\}\s*\n+def list_tools)',
            f"\n{map_line}\n" + r"\1",
            src, count=1
        )

    try:
        registry_path.write_text(src, encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, f"Не удалось записать registry.py: {e}"


# ─── основной run ────────────────────────────────────────────────────────────────────────────────────

def run(query: str, history: list[dict]) -> str:
    """
    Вход: запрос пользователя.
    Выход: строка — отчёт о создании инструмента.
    Никогда не падает без fallback.
    """
    logger.info(f"[self_extend] запрос: {query[:80]}")
    existing = list_tools()

    # ── Шаг 1: дизайн инструмента ──
    try:
        raw = chat(
            MODEL_HEAVY,
            [
                {"role": "system", "content": _DESIGN_SYSTEM.format(
                    existing_tools=", ".join(existing))},
                {"role": "user",   "content": query},
            ],
            options={"temperature": 0.1, "num_ctx": 8192},
        )
        if not raw:
            raise ValueError("пустой ответ LLM")
        raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
        raw = re.sub(r'\n?```$', '', raw).strip()
        design = json.loads(raw)
    except Exception as e:
        msg = f"Не удалось спроектировать инструмент: {e}"
        logger.error(f"[self_extend] {msg}")
        _log_result("unknown", "failed", msg, query)
        return f"Сэр, {msg}"

    tool_name  = design.get("tool_name", "").strip()
    filename   = design.get("filename", "").strip()
    entry_fn   = design.get("entry_fn", "").strip()
    description = design.get("description", query)

    if not tool_name or not filename or not entry_fn:
        msg = f"Неполный дизайн: tool_name={tool_name!r}, file={filename!r}, fn={entry_fn!r}"
        _log_result(tool_name or "?", "failed", msg, query)
        return f"Сэр, {msg}"

    if tool_name in existing:
        msg = f"Инструмент '{tool_name}' уже существует."
        _log_result(tool_name, "failed", msg, query)
        return f"Сэр, {msg}"

    # защита от path traversal
    # fix #7: заменен некорректный lstrip("tools/") на removeprefix("tools/")
    safe_filename = Path(filename).name
    norm_filename = filename.replace("\\", "/")
    inner_path = norm_filename.removeprefix("tools/")
    if not safe_filename.endswith(".py") or "/" in inner_path:
        msg = f"Недопустимый путь файла: {filename}"
        _log_result(tool_name, "security_fail", msg, query)
        return f"Сэр, {msg}"

    # ── Шаг 2: генерация кода ──
    try:
        code_raw = chat(
            MODEL_HEAVY,
            [
                {"role": "system", "content": _CODE_SYSTEM.format(
                    filename=safe_filename,
                    entry_fn=entry_fn,
                    description=description,
                )},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.05, "num_ctx": 8192},
        )
        if not code_raw:
            raise ValueError("пустой ответ LLM")
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code_raw.strip())
        code = re.sub(r'\n?```$', '', code).strip()
    except Exception as e:
        msg = f"Не удалось сгенерировать код: {e}"
        _log_result(tool_name, "failed", msg, query)
        return f"Сэр, {msg}"

    # ── Шаг 3: security + AST ──
    sec_ok, sec_msg = _security_check(code)
    if not sec_ok:
        _log_result(tool_name, "security_fail", sec_msg, query)
        return f"Сэр, отклонено по соображениям безопасности: {sec_msg}"

    ast_ok, ast_msg = _ast_check(code)
    if not ast_ok:
        _log_result(tool_name, "failed", ast_msg, query)
        return f"Сэр, код содержит синтаксические ошибки: {ast_msg}"

    # ── Шаг 4: запись файла ──
    target_path = TOOLS_DIR / safe_filename
    if target_path.exists():
        msg = f"Файл {safe_filename} уже существует, не перезаписываю."
        _log_result(tool_name, "failed", msg, query)
        return f"Сэр, {msg}"

    try:
        target_path.write_text(code, encoding="utf-8")
    except Exception as e:
        msg = f"Не удалось записать файл: {e}"
        _log_result(tool_name, "failed", msg, query)
        return f"Сэр, {msg}"

    # ── Шаг 5: патч registry ──
    reg_ok, reg_msg = _patch_registry(tool_name, safe_filename, entry_fn)
    if not reg_ok:
        target_path.unlink(missing_ok=True)  # откатываем
        _log_result(tool_name, "failed", reg_msg, query)
        return f"Сэр, файл создан, но регистрация не удалась: {reg_msg}"

    # ── Шаг 6: smoke-тест ──
    try:
        import tools.registry as _reg_mod
        importlib.reload(_reg_mod)
        from tools.registry import call_tool as _ct
        smoke = _ct(tool_name, {})
        smoke_ok = True
        smoke_detail = str(smoke.data)[:200] if smoke.ok else smoke.error
    except Exception as e:
        smoke_ok = False
        smoke_detail = str(e)

    status = "created" if smoke_ok else "smoke_fail"
    _log_result(tool_name, status, smoke_detail, query)

    if smoke_ok:
        return (
            f"Сэр, инструмент '{tool_name}' успешно создан, "
            f"зарегистрирован и прошёл smoke-тест. "
            f"Файл: tools/{safe_filename}"
        )
    else:
        return (
            f"Сэр, инструмент '{tool_name}' создан и зарегистрирован, "
            f"но smoke-тест не прошёл: {smoke_detail[:150]}. "
            f"Проверьте tools/{safe_filename} вручную."
        )
