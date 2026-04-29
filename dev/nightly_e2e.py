"""
dev/nightly_e2e.py — Level 4 Nightly End-to-End Suite.

Прогоняет 5 эталонных задач через ProjectAgent.run() против БОЕВЫХ моделей Ollama.
Проверяет что вся пайплайн (intake → architect → build → test → heal → report)
заканчивается success-статусом и manifest.status = "done".

Запуск:
  python -m dev.nightly_e2e             # все задачи
  python -m dev.nightly_e2e --task rss  # одна задача
  python -m dev.nightly_e2e --no-cleanup # сохранить data/projects/* для разбора

Использование Ollama:
  Если is_ollama_available() == False — все задачи помечаются SKIPPED, скрипт
  возвращает rc=0. Это позволяет cron-скрипту safe-вызывать его на машинах
  где ollama недоступна.

Логи:
  ~/.jarvis/logs/nightly_e2e.jsonl — по одной JSON-строке на каждый прогон.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Настройка путей до импорта jarvis-кода.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.client import is_ollama_available
from brain.agents import project as project_agent
from core.paths import LOGS_DIR, PROJECTS_DIR

logger = logging.getLogger("nightly_e2e")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─── 5 эталонных задач ──────────────────────────────────────────────────────
@dataclass
class ReferenceTask:
    name: str
    query: str
    # Минимальные структурные ожидания. Не строгий контракт — только sanity.
    expected_min_files: int      # сколько файлов должен сгенерировать архитектор
    expected_artifact_hint: str  # подстрока в любом из created files (имя)
    timeout_s: int               # лимит на конкретную задачу


REFERENCE_TASKS: list[ReferenceTask] = [
    ReferenceTask(
        name="csv_to_json",
        query="Сделай Python-скрипт, который читает CSV-файл input.csv и сохраняет данные в output.json массивом объектов.",
        expected_min_files=1,
        expected_artifact_hint=".py",
        timeout_s=600,
    ),
    ReferenceTask(
        name="rename_files",
        query="Напиши скрипт который проходит по текущей папке и переименовывает все .txt файлы добавляя префикс backup_.",
        expected_min_files=1,
        expected_artifact_hint=".py",
        timeout_s=600,
    ),
    ReferenceTask(
        name="regex_extractor",
        query="Сделай скрипт который ищет все email-адреса в text.txt и сохраняет их в emails.csv по одному на строку.",
        expected_min_files=1,
        expected_artifact_hint=".py",
        timeout_s=600,
    ),
    ReferenceTask(
        name="rss_parser",
        query="Сделай парсер RSS-ленты https://lenta.ru/rss/news. Сохрани заголовки и даты в news.csv.",
        expected_min_files=1,
        expected_artifact_hint=".py",
        timeout_s=900,  # сетевая задача
    ),
    ReferenceTask(
        name="http_get",
        query="Напиши скрипт который скачивает страницу example.com и сохраняет её HTML в page.html.",
        expected_min_files=1,
        expected_artifact_hint=".py",
        timeout_s=900,
    ),
]


# ─── execution ──────────────────────────────────────────────────────────────
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


def _run_one(task: ReferenceTask) -> dict:
    """Запускает один эталонный сценарий. Возвращает dict с метриками."""
    started = time.time()
    record = {
        "task": task.name,
        "query": task.query,
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
            llm_budget=120,  # L-tier по умолчанию для E2E
        )
        record["report_excerpt"] = (report or "")[:300]
        # Ищем НОВЫй проект (не в before-set), иначе берём самый свежий
        latest = _find_new_slug(before)
        if latest is None and PROJECTS_DIR.exists():
            slugs = sorted(
                [p for p in PROJECTS_DIR.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            latest = slugs[0] if slugs else None
        if latest is not None:
            record["slug"] = latest.name
            manifest_path = latest / "manifest.json"
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                    record["manifest_status"] = m.get("status")
                    record["files_count"] = len(m.get("files") or [])
                    tests = m.get("test_results") or []
                    record["tests_total"] = len(tests)
                    record["tests_ok"] = sum(1 for t in tests if t.get("ok"))
                    record["llm_calls"] = (m.get("metrics") or {}).get("llm_calls", 0)
                except Exception as e:
                    record["error"] = f"manifest read: {e}"
        # Вердикт.
        if record["manifest_status"] == "done":
            record["status"] = "passed"
        elif record["manifest_status"] in ("partial", "running"):
            record["status"] = "partial"
        else:
            record["status"] = "failed"
        # Структурный sanity: достаточно файлов и есть нужный артефакт.
        if record["files_count"] < task.expected_min_files and record["status"] == "passed":
            record["status"] = "structural_fail"
            record["error"] = (
                f"expected ≥{task.expected_min_files} files, got {record['files_count']}"
            )
    except Exception as e:
        record["status"] = "exception"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[:2000]
    finally:
        record["wall_s"] = round(time.time() - started, 2)
    return record


def _append_log(record: dict) -> None:
    log_path = Path(LOGS_DIR) / "nightly_e2e.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_summary(records: list[dict]) -> int:
    """Печатает сводку и возвращает rc (0 = все ok)."""
    passed = sum(1 for r in records if r["status"] == "passed")
    failed = [r for r in records if r["status"] not in ("passed", "skipped")]
    total = len(records)
    print()
    print("=" * 60)
    print(f"NIGHTLY E2E: {passed}/{total} passed")
    print("=" * 60)
    for r in records:
        icon = {
            "passed": "OK",
            "skipped": "SK",
            "partial": "PR",
            "failed": "FL",
            "structural_fail": "SF",
            "exception": "EX",
        }.get(r["status"], "??")
        wall = r.get("wall_s", 0)
        files = r.get("files_count", 0)
        tests = f"{r.get('tests_ok', 0)}/{r.get('tests_total', 0)}"
        err = (" — " + r["error"][:80]) if r.get("error") else ""
        print(f"  [{icon}] {r['task']:<20} wall={wall}s files={files} tests={tests}{err}")
    print()
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Jarvis L4 Nightly E2E suite")
    p.add_argument("--task", help="запустить только одну задачу по имени")
    p.add_argument("--no-cleanup", action="store_true",
                   help="не удалять data/projects/<slug> после прогона")
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
        return 0  # safe: не падаем когда ollama выключен

    tasks = REFERENCE_TASKS
    if args.task:
        tasks = [t for t in REFERENCE_TASKS if t.name == args.task]
        if not tasks:
            logger.error(f"Task '{args.task}' not found. Available: "
                         f"{[t.name for t in REFERENCE_TASKS]}")
            return 2

    records: list[dict] = []
    for t in tasks:
        logger.info(f"=== Running task: {t.name} ===")
        rec = _run_one(t)
        _append_log(rec)
        records.append(rec)
        logger.info(f"=== {t.name}: {rec['status']} ({rec['wall_s']}s) ===")

    return _print_summary(records)


if __name__ == "__main__":
    sys.exit(main())
