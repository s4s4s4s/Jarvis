"""
dev/multifile_diag.py — P11.0 Multi-file Diagnostic Suite.

Цель: понять где ломается ProjectAgent на multi-file проектах ДО того, как
писать фиксы P11.1+. Прогоняет 1-3 эталонных multi-file задачи с расширенным
бюджетом и собирает РАСШИРЕННУЮ диагностику в JSON-отчёт:

  - какие файлы создались / провалились
  - есть ли cross-file imports и AttributeError'ы в test stderr
  - сколько раз heal-loop проходил без прогресса
  - какие чеки реально прошли, какие фильтр срезал
  - что в plan.files (есть ли depends_on заполнен)
  - сколько байт каждый файл; нет ли в коде phantom-импортов

ЗАПУСК (Windows):
  cd C:\\Jarvis
  python -m dev.multifile_diag                       # все задачи
  python -m dev.multifile_diag --task tg_bot_lite    # одна
  python -m dev.multifile_diag --no-cleanup          # сохранить data/projects/

Логи:
  ~/.jarvis/logs/multifile_diag.jsonl   — JSON-строка на прогон
  ~/.jarvis/logs/multifile_diag_<slug>.report.txt — человекочитаемый отчёт

Принципы:
- Локально, без облачных API.
- НЕ ставит assertions: задача ДИАГНОСТИЧЕСКАЯ. Цель — увидеть где ломается.
- Все 3 задачи в каталоге (идём по возрастанию сложности): 2 файла → 4 → 6.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Настройка путей до импорта jarvis-кода.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.client import is_ollama_available
from brain.agents import project as project_agent
from core.paths import LOGS_DIR, PROJECTS_DIR

logger = logging.getLogger("multifile_diag")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─── 3 эталонные multi-file задачи (по возрастанию сложности) ──────────────
@dataclass
class MultiFileTask:
    name: str
    query: str
    expected_min_files: int       # минимум файлов которые должен создать архитектор
    expected_modules: list[str]   # ожидаемые роли модулей (для report-чек-листа)
    timeout_s: int                # лимит wall-clock на задачу
    llm_budget: int               # лимит LLM-вызовов

    # Подсказки для пост-анализа (НЕ влияет на pass/fail — только в отчёт):
    expected_imports: list[str] = field(default_factory=list)
    expected_calls:   list[str] = field(default_factory=list)


# Все задачи спроектированы так, чтобы НЕ требовать сети и внешних API
# (соответствует принципу «всё локально»). Все «боты» работают на stdin/stdout
# или плоских файлах, имитируя реальную архитектуру без сетевой зависимости.
REFERENCE_TASKS: list[MultiFileTask] = [
    MultiFileTask(
        name="calculator_split",
        query=(
            "Сделай Python-проект из двух файлов: "
            "operations.py содержит функции add(a,b), sub(a,b), mul(a,b), div(a,b) "
            "(каждая возвращает число, div выбрасывает ValueError при делении на ноль); "
            "main.py импортирует эти функции и при запуске печатает результат "
            "вызова всех четырёх с числами 6 и 2."
        ),
        expected_min_files=2,
        expected_modules=["operations", "main"],
        expected_imports=["from operations import"],
        expected_calls=["add(6,", "sub(6,", "mul(6,", "div(6,"],
        timeout_s=900,   # 15 мин
        llm_budget=60,
    ),
    MultiFileTask(
        name="todo_storage_lite",
        query=(
            "Сделай простой проект todo-list из трёх файлов: "
            "storage.py — модуль работы с файлом todos.json, экспортирует функции "
            "load_todos() (возвращает list[dict]), save_todos(items: list[dict]), "
            "add_todo(text: str) и list_todos() (печатает все на экран); "
            "cli.py — функция main() парсит sys.argv и вызывает add_todo или list_todos; "
            "main.py — точка входа, вызывает cli.main(). "
            "Скрипт `python main.py add hello` должен сохранить запись в todos.json. "
            "Скрипт `python main.py list` — вывести её. Используй stdlib (json, sys, os)."
        ),
        expected_min_files=3,
        expected_modules=["storage", "cli", "main"],
        expected_imports=["from storage import", "from cli import", "import storage", "import cli"],
        expected_calls=["add_todo(", "list_todos(", "save_todos(", "load_todos("],
        timeout_s=1200,  # 20 мин
        llm_budget=80,
    ),
    MultiFileTask(
        name="tg_reminder_bot_lite",
        query=(
            "Сделай локальный проект «бот напоминаний» (БЕЗ настоящего Telegram, "
            "общается через stdin/stdout) из четырёх файлов: "
            "config.py — переменные DB_PATH='reminders.sqlite' и DEFAULT_USER='local'; "
            "storage.py — функции init_db(), add_reminder(user, text, when_iso), "
            "list_reminders(user) (использует sqlite3 из stdlib, схема: id, user, text, when_iso); "
            "bot.py — функция handle_command(line: str) -> str парсит команды "
            "'/add <text> @ <iso>' и '/list', возвращает текст ответа; "
            "main.py — точка входа: init_db(), читает строки из stdin до EOF, "
            "печатает ответы handle_command. "
            "При запуске `echo /list | python main.py` должна напечататься пустая или "
            "непустая сводка без падений. Используй только stdlib (sqlite3, sys, datetime)."
        ),
        expected_min_files=4,
        expected_modules=["config", "storage", "bot", "main"],
        expected_imports=[
            "from config import", "from storage import", "from bot import",
            "import config", "import storage", "import bot",
        ],
        expected_calls=["init_db(", "add_reminder(", "list_reminders(", "handle_command("],
        timeout_s=1500,  # 25 мин
        llm_budget=100,
    ),
]


# ─── Diagnostic helpers ────────────────────────────────────────────────────
def _snapshot_existing_slugs() -> set[str]:
    if not PROJECTS_DIR.exists():
        return set()
    return {p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()}


def _find_new_slug(before: set[str]) -> Path | None:
    if not PROJECTS_DIR.exists():
        return None
    after = [(p, p.stat().st_mtime) for p in PROJECTS_DIR.iterdir()
             if p.is_dir() and p.name not in before]
    if not after:
        return None
    after.sort(key=lambda x: x[1], reverse=True)
    return after[0][0]


def _read_project_files(proj_dir: Path) -> dict[str, str]:
    """Читает все .py файлы в корне проекта (без venv/__pycache__)."""
    out: dict[str, str] = {}
    if not proj_dir.exists():
        return out
    for p in proj_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(".") or name in {"manifest.json", "REPORT.md", "report.md"}:
            continue
        if not (name.endswith(".py") or name.endswith(".txt") or name.endswith(".md")):
            continue
        try:
            out[name] = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return out


def _collect_imports(files: dict[str, str]) -> dict[str, list[str]]:
    """Извлекает import statements из каждого .py файла. Грубо, без AST."""
    imp_re = re.compile(r"^\s*(from\s+[\w\.]+\s+import\s+[^\n]+|import\s+[\w\.,\s]+)$",
                        re.MULTILINE)
    out: dict[str, list[str]] = {}
    for name, content in files.items():
        if not name.endswith(".py"):
            continue
        out[name] = [m.group(0).strip() for m in imp_re.finditer(content)]
    return out


def _collect_defs(files: dict[str, str]) -> dict[str, list[str]]:
    """Извлекает def/class сигнатуры (грубо)."""
    sig_re = re.compile(r"^(def\s+\w+\s*\([^)]*\)|class\s+\w+\s*(?:\([^)]*\))?\s*:)",
                        re.MULTILINE)
    out: dict[str, list[str]] = {}
    for name, content in files.items():
        if not name.endswith(".py"):
            continue
        out[name] = [m.group(0).strip().rstrip(":") for m in sig_re.finditer(content)]
    return out


def _analyze_phases(phases: list[dict]) -> dict:
    """Анализирует phases manifest'a: считает heal-итерации, импорт-ошибки в stderr."""
    stats = {
        "total": len(phases),
        "ok": sum(1 for p in phases if p.get("status") == "ok"),
        "failed": sum(1 for p in phases if p.get("status") not in ("ok", None)),
        "build_phases": [],
        "test_phases": [],
        "heal_iters": 0,
        "import_errors": [],     # cross-file ImportError/ModuleNotFoundError
        "attribute_errors": [],  # AttributeError
        "stderr_excerpts": [],   # первые 3 ненулевых stderr
    }
    for p in phases:
        nm = str(p.get("name", ""))
        if nm.startswith("build:"):
            stats["build_phases"].append({"name": nm, "status": p.get("status")})
        elif nm.startswith("test:"):
            stats["test_phases"].append({"name": nm, "status": p.get("status")})
        elif nm.startswith("heal"):
            stats["heal_iters"] += 1
        # Парсим все известные места где может лежать stderr/traceback:
        # details и detail — свободный текст; stderr — отдельное поле фазы;
        # result — вложенный dict от _run_one_test с stdout/stderr/checks.
        det_parts = [
            str(p.get("details") or ""),
            str(p.get("detail") or ""),
            str(p.get("stderr") or ""),
        ]
        result = p.get("result")
        if isinstance(result, dict):
            det_parts.append(str(result.get("stderr") or ""))
            det_parts.append(str(result.get("stdout") or ""))
        det = "\n".join(s for s in det_parts if s)
        if not det:
            continue
        if "ImportError" in det or "ModuleNotFoundError" in det:
            m = re.search(r"(?:Import|Module(?:NotFound))?Error[^\n]{0,200}", det)
            stats["import_errors"].append({"phase": nm, "msg": (m.group(0) if m else det[:200])})
        if "AttributeError" in det:
            m = re.search(r"AttributeError[^\n]{0,200}", det)
            stats["attribute_errors"].append({"phase": nm, "msg": (m.group(0) if m else det[:200])})
        if len(stats["stderr_excerpts"]) < 3 and ("Traceback" in det or "Error" in det):
            stats["stderr_excerpts"].append({"phase": nm, "excerpt": det[:400]})
    return stats


def _analyze_plan(plan: dict | None) -> dict:
    """Анализ архитектурного плана: сколько файлов, есть ли depends_on, exports."""
    if not isinstance(plan, dict):
        return {"present": False}
    files = plan.get("files") or []
    out = {
        "present": True,
        "files_count": len(files),
        "files_with_depends_on": 0,
        "files_with_exports": 0,
        "files_with_purpose": 0,
        "depends_on_map": {},
        "exports_per_file": {},   # P11.1: path -> list of export names
        "inputs_count": len(plan.get("inputs") or []),
        "tests_count": len(plan.get("tests") or []),
        "build_steps_count": len(plan.get("build_steps") or []),
        "contract_metrics": plan.get("_contract_metrics") or {},  # P11.1
    }
    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path", "?")
        if f.get("depends_on"):
            out["files_with_depends_on"] += 1
            out["depends_on_map"][path] = f.get("depends_on")
        exports = f.get("exports") or []
        if exports:
            out["files_with_exports"] += 1
            # P11.1: вытаскиваем имена из dict-элементов или строк
            names = []
            for e in exports:
                if isinstance(e, dict):
                    nm = e.get("name")
                    if nm:
                        names.append(str(nm))
                elif isinstance(e, str):
                    names.append(e)
            if names:
                out["exports_per_file"][path] = names
        if f.get("purpose"):
            out["files_with_purpose"] += 1
    return out


def _check_expected(task: MultiFileTask,
                    files: dict[str, str],
                    imports: dict[str, list[str]]) -> dict:
    """Проверяет какие из ожиданий task реально выполнились."""
    out = {
        "modules_present": {},
        "imports_found": {},
        "calls_found": {},
    }
    # модули (имена .py файлов без расширения)
    actual_modules = {n[:-3] for n in files.keys() if n.endswith(".py")}
    for mod in task.expected_modules:
        out["modules_present"][mod] = mod in actual_modules
    # cross-file импорты
    all_imports = " ".join(line for lines in imports.values() for line in lines)
    for hint in task.expected_imports:
        out["imports_found"][hint] = hint in all_imports
    # вызовы
    all_code = " ".join(files.values())
    for call in task.expected_calls:
        out["calls_found"][call] = call in all_code
    return out


# ─── Run one task ──────────────────────────────────────────────────────────
def _run_one(task: MultiFileTask) -> dict:
    started = time.time()
    record: dict = {
        "task": task.name,
        "query": task.query[:200],
        "started_at": datetime.utcnow().isoformat() + "Z",
        "status": "unknown",
        "wall_s": 0.0,
        "manifest_status": None,
        "files_count": 0,
        "tests_total": 0,
        "tests_ok": 0,
        "llm_calls": 0,
        "error": "",
        "report_excerpt": "",
        "slug": "",
    }
    before = _snapshot_existing_slugs()
    try:
        report = project_agent.run(
            task.query,
            history=None,
            wall_budget_s=task.timeout_s,
            llm_budget=task.llm_budget,
        )
        record["report_excerpt"] = (report or "")[:500]
        latest = _find_new_slug(before)
        if latest is None and PROJECTS_DIR.exists():
            slugs = sorted([p for p in PROJECTS_DIR.iterdir() if p.is_dir()],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            latest = slugs[0] if slugs else None
        if latest is not None:
            record["slug"] = latest.name
            manifest_path = latest / "manifest.json"
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception as e:
                    record["error"] = f"manifest read: {e}"
                    m = {}
                record["manifest_status"] = m.get("status")
                phases = m.get("phases") or []
                # P11.0 fix (FM-8): files_count — реально созданные кодовые файлы на диске,
                # без README.md/dotfiles/logs. Раньше брали длину списка из manifest —
                # там хранится план архитектора, но факт может отличаться.
                _disk_files = _read_project_files(latest)
                _code_files = [n for n in _disk_files.keys()
                               if not n.startswith(".")
                               and n.lower() != "readme.md"
                               and not n.startswith("logs/")
                               and not n.startswith(".venv/")]
                record["files_count"] = len(_code_files)
                record["tests_total"] = len([p for p in phases
                                              if str(p.get("name", "")).startswith("test:")])
                record["tests_ok"] = sum(1 for p in phases
                                          if str(p.get("name", "")).startswith("test:")
                                          and p.get("status") == "ok")
                record["llm_calls"] = (m.get("metrics") or {}).get("llm_used", 0)

                # P11.0 расширенная диагностика:
                record["plan_analysis"] = _analyze_plan(m.get("plan"))
                record["phase_stats"] = _analyze_phases(phases)
                files = _disk_files
                record["files_on_disk"] = sorted(files.keys())
                record["file_sizes"] = {n: len(c) for n, c in files.items()}
                record["imports_per_file"] = _collect_imports(files)
                record["defs_per_file"] = _collect_defs(files)
                record["expected_check"] = _check_expected(task, files,
                                                            record["imports_per_file"])

        # Вердикт.
        if record["manifest_status"] == "done":
            record["status"] = "passed"
        elif record["manifest_status"] in ("partial", "running"):
            record["status"] = "partial"
        else:
            record["status"] = "failed"

        if record["files_count"] < task.expected_min_files and record["status"] == "passed":
            record["status"] = "structural_fail"
            record["error"] = (
                f"expected >={task.expected_min_files} files, got {record['files_count']}"
            )
    except Exception as e:
        record["status"] = "exception"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[:2000]
    finally:
        record["wall_s"] = round(time.time() - started, 2)
    return record


def _append_log(record: dict) -> None:
    log_path = Path(LOGS_DIR) / "multifile_diag.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_human_report(record: dict) -> Path | None:
    """Пишет человекочитаемый отчёт. Возвращает путь или None."""
    slug = record.get("slug") or record.get("task")
    if not slug:
        return None
    path = Path(LOGS_DIR) / f"multifile_diag_{slug}.report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# P11.0 Multi-file Diagnostic Report — {record.get('task')}")
    lines.append(f"slug: {record.get('slug', '?')}")
    lines.append(f"status: {record.get('status')}    "
                 f"manifest={record.get('manifest_status')}    "
                 f"wall={record.get('wall_s')}s    "
                 f"llm_calls={record.get('llm_calls')}")
    lines.append("")
    lines.append("## PLAN ANALYSIS")
    pa = record.get("plan_analysis") or {}
    if pa.get("present"):
        lines.append(f"  files_count:           {pa.get('files_count')}")
        lines.append(f"  files_with_purpose:    {pa.get('files_with_purpose')}")
        lines.append(f"  files_with_depends_on: {pa.get('files_with_depends_on')}")
        lines.append(f"  files_with_exports:    {pa.get('files_with_exports')}")
        lines.append(f"  inputs:                {pa.get('inputs_count')}")
        lines.append(f"  tests:                 {pa.get('tests_count')}")
        depmap = pa.get("depends_on_map") or {}
        if depmap:
            lines.append("  depends_on_map:")
            for k, v in depmap.items():
                lines.append(f"    {k}: {v}")
        # P11.1: exports per file + contract metrics
        exp_map = pa.get("exports_per_file") or {}
        if exp_map:
            lines.append("  exports_per_file:")
            for k, names in exp_map.items():
                lines.append(f"    {k}: {names}")
        cm = pa.get("contract_metrics") or {}
        if cm:
            lines.append("  contract_metrics:")
            lines.append(f"    files_total:           {cm.get('files_total')}")
            lines.append(f"    py_files:              {cm.get('py_files')}")
            lines.append(f"    files_with_exports:    {cm.get('files_with_exports')}")
            mis = cm.get("files_missing_exports") or []
            if mis:
                lines.append(f"    files_missing_exports: {mis}")
            unmatched = cm.get("depends_unmatched") or []
            if unmatched:
                lines.append(f"    depends_unmatched:     {unmatched}")
            outside = cm.get("depends_outside_plan") or []
            if outside:
                lines.append(f"    depends_outside_plan:  {outside}")
    else:
        lines.append("  (plan отсутствует в manifest)")
    lines.append("")
    lines.append("## PHASES STATS")
    ps = record.get("phase_stats") or {}
    lines.append(f"  total: {ps.get('total')}    ok: {ps.get('ok')}    failed: {ps.get('failed')}")
    lines.append(f"  build phases: {len(ps.get('build_phases') or [])}")
    for bp in (ps.get("build_phases") or []):
        lines.append(f"    [{bp.get('status')}] {bp.get('name')}")
    lines.append(f"  test phases: {len(ps.get('test_phases') or [])}")
    for tp in (ps.get("test_phases") or []):
        lines.append(f"    [{tp.get('status')}] {tp.get('name')}")
    lines.append(f"  heal_iters: {ps.get('heal_iters')}")
    if ps.get("import_errors"):
        lines.append(f"  IMPORT ERRORS ({len(ps['import_errors'])}):")
        for ie in ps["import_errors"][:5]:
            lines.append(f"    [{ie.get('phase')}] {ie.get('msg')}")
    if ps.get("attribute_errors"):
        lines.append(f"  ATTRIBUTE ERRORS ({len(ps['attribute_errors'])}):")
        for ae in ps["attribute_errors"][:5]:
            lines.append(f"    [{ae.get('phase')}] {ae.get('msg')}")
    if ps.get("stderr_excerpts"):
        lines.append("  STDERR EXCERPTS:")
        for ex in ps["stderr_excerpts"]:
            lines.append(f"    [{ex.get('phase')}] {ex.get('excerpt')}")
    lines.append("")
    lines.append("## FILES ON DISK")
    fs = record.get("file_sizes") or {}
    for name in sorted(fs.keys()):
        lines.append(f"  {name:<30} {fs[name]:>6} bytes")
    lines.append("")
    lines.append("## IMPORTS")
    for name, imps in (record.get("imports_per_file") or {}).items():
        lines.append(f"  {name}:")
        for imp in imps:
            lines.append(f"    {imp}")
    lines.append("")
    lines.append("## DEFINITIONS")
    for name, defs in (record.get("defs_per_file") or {}).items():
        lines.append(f"  {name}:")
        for d in defs:
            lines.append(f"    {d}")
    lines.append("")
    lines.append("## EXPECTED CHECKS")
    ec = record.get("expected_check") or {}
    mp = ec.get("modules_present") or {}
    if mp:
        lines.append("  modules:")
        for k, v in mp.items():
            mark = "OK" if v else "MISSING"
            lines.append(f"    [{mark}] {k}.py")
    ifd = ec.get("imports_found") or {}
    if ifd:
        lines.append("  imports:")
        for k, v in ifd.items():
            mark = "OK" if v else "MISSING"
            lines.append(f"    [{mark}] {k}")
    cf = ec.get("calls_found") or {}
    if cf:
        lines.append("  calls:")
        for k, v in cf.items():
            mark = "OK" if v else "MISSING"
            lines.append(f"    [{mark}] {k}")
    lines.append("")
    if record.get("error"):
        lines.append(f"## ERROR\n  {record['error']}")
    if record.get("report_excerpt"):
        lines.append("\n## AGENT REPORT (excerpt)")
        lines.append(record["report_excerpt"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_summary(records: list[dict]) -> int:
    passed = sum(1 for r in records if r["status"] == "passed")
    total = len(records)
    print()
    # FM-6 safe print: на Windows stdout = cp1251, любой юникод символ
    # выходящий за рамки cp1251 вырубит весь summary. Изолируем.
    def _safe(s: str) -> str:
        try:
            enc = (sys.stdout.encoding or "utf-8").lower()
            return s.encode(enc, errors="replace").decode(enc, errors="replace")
        except Exception:
            return s
    def _p(s: str) -> None:
        print(_safe(s))
    _p("=" * 72)
    _p(f"P11.0 MULTI-FILE DIAG: {passed}/{total} passed")
    _p("=" * 72)
    for r in records:
        icon = {
            "passed": "OK", "skipped": "SK", "partial": "PR",
            "failed": "FL", "structural_fail": "SF", "exception": "EX",
        }.get(r["status"], "??")
        wall = r.get("wall_s", 0)
        files = r.get("files_count", 0)
        tests = f"{r.get('tests_ok', 0)}/{r.get('tests_total', 0)}"
        ms = r.get("manifest_status") or "-"
        ec = r.get("expected_check") or {}
        miss_mods = [k for k, v in (ec.get("modules_present") or {}).items() if not v]
        miss_imps = [k for k, v in (ec.get("imports_found") or {}).items() if not v]
        miss_str = ""
        if miss_mods:
            miss_str += f" missing_modules={','.join(miss_mods)}"
        if miss_imps:
            miss_str += f" missing_imports={len(miss_imps)}"
        ps = r.get("phase_stats") or {}
        if ps.get("import_errors"):
            miss_str += f" import_errs={len(ps['import_errors'])}"
        if ps.get("attribute_errors"):
            miss_str += f" attr_errs={len(ps['attribute_errors'])}"
        err = (" -- " + (r.get("error") or "")[:80]) if r.get("error") else ""
        _p(f"  [{icon}] {r['task']:<22} wall={wall}s files={files} "
           f"tests={tests} manifest={ms}{miss_str}{err}")
        rep = r.get("_report_path")
        if rep:
            _p(f"          report: {rep}")
    _p("")
    return 0  # P11.0: НИКОГДА не падаем — это диагностика, не CI-гейт


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Jarvis L4-P11.0 Multi-file diagnostic suite")
    p.add_argument("--task", help="запустить только одну задачу по имени")
    p.add_argument("--no-cleanup", action="store_true",
                   help="не удалять data/projects/<slug> (по умолчанию НЕ удаляем)")
    args = p.parse_args(argv)

    if not is_ollama_available():
        logger.warning("Ollama недоступна — все задачи SKIPPED")
        records = [
            {"task": t.name, "status": "skipped", "error": "ollama unavailable"}
            for t in REFERENCE_TASKS
        ]
        for r in records:
            _append_log(r)
        _print_summary(records)
        return 0

    tasks = REFERENCE_TASKS
    if args.task:
        tasks = [t for t in REFERENCE_TASKS if t.name == args.task]
        if not tasks:
            logger.error(f"Task '{args.task}' not found. Available: "
                         f"{[t.name for t in REFERENCE_TASKS]}")
            return 2

    records: list[dict] = []
    for t in tasks:
        logger.info(f"=== P11.0 task: {t.name} (timeout={t.timeout_s}s, llm_budget={t.llm_budget}) ===")
        rec = _run_one(t)
        # пишем human-readable отчёт
        rep_path = _write_human_report(rec)
        if rep_path:
            rec["_report_path"] = str(rep_path)
        _append_log(rec)
        records.append(rec)
        logger.info(f"=== {t.name}: {rec['status']} ({rec['wall_s']}s) ===")

    return _print_summary(records)


if __name__ == "__main__":
    sys.exit(main())
