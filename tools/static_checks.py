"""
tools/static_checks.py — детерминистические статанализаторы для build-loop.

Цель: ловить синтаксические и тривиальные ошибки ДО вызова LLM-Reviewer,
экономить LLM-вызовы и быстро возвращать понятный feedback в Coder.

Используется в brain/agents/project.py::_build_one_file.

Дизайн:
  - Всё локально, никаких облачных API.
  - Если внешние линтеры (ruff/pyflakes) недоступны — fallback на ast.parse.
  - Никогда не падает: при любой ошибке возвращает структурированный dict.
  - Обрабатывает ТОЛЬКО Python (.py). Для других расширений — ok=True, no checks.

API:
  ast_check(code) -> {"ok": bool, "errors": [str], "tool": "ast"}
  lint_check(code) -> {"ok": bool, "warnings": [str], "tool": "ruff"|"pyflakes"|"none"}
  static_check(path, code) -> объединённый результат: ok, errors, warnings, tools
"""
from __future__ import annotations

import ast
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Сколько секунд на запуск внешнего линтера.
LINT_TIMEOUT_S = 10
# Максимум сообщений, которые передаём в feedback (чтобы не раздувать промт).
MAX_MESSAGES = 8


def _is_python(path: str) -> bool:
    return isinstance(path, str) and path.lower().endswith(".py")


def ast_check(code: str) -> dict[str, Any]:
    """Проверяет, что код синтаксически валиден через ast.parse.

    Возвращает:
        {"ok": True,  "errors": [],          "tool": "ast"}     если ok
        {"ok": False, "errors": ["..."],     "tool": "ast"}     если SyntaxError
    """
    if not isinstance(code, str):
        return {"ok": False, "errors": ["code is not a string"], "tool": "ast"}
    if not code.strip():
        return {"ok": False, "errors": ["empty code"], "tool": "ast"}
    try:
        ast.parse(code)
        return {"ok": True, "errors": [], "tool": "ast"}
    except SyntaxError as e:
        # Формат: "L<line>:<col> SyntaxError: <msg>" — однозначно для Coder.
        line = e.lineno or 0
        col = e.offset or 0
        msg = e.msg or "syntax error"
        text = (e.text or "").rstrip()
        detail = f"L{line}:{col} SyntaxError: {msg}"
        if text:
            detail += f"   |   near: {text!r}"
        return {"ok": False, "errors": [detail], "tool": "ast"}
    except (ValueError, TypeError) as e:
        # ast.parse может бросить ValueError на null-байтах и т.п.
        return {"ok": False, "errors": [f"parse error: {e}"], "tool": "ast"}


def _have(tool: str) -> bool:
    """Есть ли команда tool в PATH или это импортируемый модуль."""
    if shutil.which(tool):
        return True
    # ruff/pyflakes часто лежат в site-packages, доступны как python -m
    try:
        import importlib
        importlib.import_module(tool)
        return True
    except Exception:
        return False


def _run_lint(cmd: list[str], code: str) -> dict[str, Any]:
    """Запускает линтер на временном файле и возвращает stdout/rc."""
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="jarvis-lint-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        proc = subprocess.run(
            [*cmd, tmp_path],
            capture_output=True,
            text=True,
            timeout=LINT_TIMEOUT_S,
        )
        return {
            "rc": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {"rc": -1, "stdout": "", "stderr": "lint timeout"}
    except FileNotFoundError as e:
        return {"rc": -1, "stdout": "", "stderr": f"lint not found: {e}"}
    except Exception as e:
        return {"rc": -1, "stdout": "", "stderr": f"lint error: {e}"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _parse_lint_output(stdout: str, tmp_path_hint: str = "") -> list[str]:
    """Превращает многострочный вывод линтера в список коротких сообщений."""
    out: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # ruff: <path>:<line>:<col>: <code> <msg>
        # pyflakes: <path>:<line>:<col>: <msg>
        # отрезаем абсолютный путь временного файла
        idx = line.find(":")
        if idx > 0 and tmp_path_hint and tmp_path_hint in line:
            line = line.split(tmp_path_hint, 1)[-1].lstrip(":")
        # фильтруем шум вроде "Found N errors."
        low = line.lower()
        if low.startswith("found ") or low.startswith("all checks passed"):
            continue
        out.append(line)
        if len(out) >= MAX_MESSAGES:
            break
    return out


def lint_check(code: str) -> dict[str, Any]:
    """Запускает ruff (если есть), иначе pyflakes, иначе пропускает.

    Никогда не возвращает ok=False — это soft-проверка для подсказок,
    а не блокировка. Падающие линтеры → tool='none', warnings=[].
    """
    if not isinstance(code, str) or not code.strip():
        return {"ok": True, "warnings": [], "tool": "none"}

    # Пробуем ruff
    if _have("ruff"):
        # ruff check --output-format=concise <file>
        res = _run_lint(["ruff", "check", "--output-format=concise"], code)
        # ruff: rc=0 → нет проблем, rc=1 → есть warnings, rc>1 → ошибка запуска
        if res["rc"] in (0, 1):
            warnings = _parse_lint_output(res["stdout"])
            return {"ok": True, "warnings": warnings, "tool": "ruff"}
        # rc>1 → fallthrough на pyflakes

    # Пробуем pyflakes (через python -m, чтобы не зависеть от shim в PATH)
    try:
        import pyflakes  # noqa: F401
        import sys
        res = _run_lint([sys.executable, "-m", "pyflakes"], code)
        if res["rc"] in (0, 1):
            warnings = _parse_lint_output(res["stdout"])
            return {"ok": True, "warnings": warnings, "tool": "pyflakes"}
    except ImportError:
        pass

    return {"ok": True, "warnings": [], "tool": "none"}


def static_check(path: str, code: str) -> dict[str, Any]:
    """Объединённая проверка: ast (блокирует) + lint (soft warnings).

    Returns:
        {
            "ok":        bool,        # False ТОЛЬКО если ast упал
            "errors":    [str],       # из ast
            "warnings":  [str],       # из lint
            "tools":     [str],       # какие инструменты сработали
            "applicable": bool,       # был ли это Python-файл
        }
    """
    if not _is_python(path):
        return {"ok": True, "errors": [], "warnings": [], "tools": [], "applicable": False}

    ast_res = ast_check(code)
    if not ast_res["ok"]:
        # Синтаксис битый — линт смысла не имеет.
        return {
            "ok": False,
            "errors": ast_res["errors"],
            "warnings": [],
            "tools": ["ast"],
            "applicable": True,
        }

    lint_res = lint_check(code)
    return {
        "ok": True,
        "errors": [],
        "warnings": lint_res["warnings"],
        "tools": ["ast", lint_res["tool"]],
        "applicable": True,
    }


def static_errors_to_feedback(errors: list[str]) -> str:
    """Форматирует ast-ошибки в строку, которую Coder получит как feedback."""
    if not errors:
        return ""
    head = "Код синтаксически невалиден. Исправь следующие ошибки:"
    bullets = "\n".join(f"  - {e}" for e in errors[:MAX_MESSAGES])
    return f"{head}\n{bullets}"


def static_warnings_to_hint(warnings: list[str]) -> str:
    """Форматирует lint-warnings как мягкую подсказку (не feedback, а hint)."""
    if not warnings:
        return ""
    head = "Статический анализ нашёл потенциальные проблемы (не критично):"
    bullets = "\n".join(f"  - {w}" for w in warnings[:MAX_MESSAGES])
    return f"{head}\n{bullets}"
