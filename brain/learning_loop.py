# brain/learning_loop.py
"""
Петля самообучения Jarvis.

Что делает:
1. Читает logs/feedback.jsonl
2. Верифицированные success → добавляет в route_examples.jsonl
3. Верифицированные failure → удаляет совпадающие auto-примеры
4. auto-success с confidence ≥ 0.90 → добавляет без верификации
5. Сбрасывает embed-кэш, архивирует обработанные записи

fix N7/N1: весь цикл load→modify→save выполняется атомарно под _lock
через _run_with_lock(). Устраняет race condition между run_learning_cycle()
и remove_failed_example() при параллельных вызовах из ThreadPoolExecutor.

fix BUG-4: remove_failed_example теперь делает load_all() и поиск записи
ВНУТРИ _run_with_lock, полностью устраняя TOCTOU race condition, при котором
archive_processed() мог удалить запись между load_all() и _run_with_lock(),
что приводило к удалению ВСЕХ auto_success/learned_success примеров.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from core.paths import ROUTE_EXAMPLES, LEARNING_REPORT

logger = logging.getLogger(__name__)
_lock = threading.Lock()

AUTO_SUCCESS_THRESHOLD = 0.90  # confidence ниже этого → не добавляем


# ─── атомарный helper ────────────────────────────────────────────────────────

def _read_examples_locked() -> list[dict]:
    """Читает route_examples.jsonl. Вызывается ТОЛЬКО внутри _lock."""
    if not ROUTE_EXAMPLES.exists():
        return []
    result = []
    for line in ROUTE_EXAMPLES.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def _write_examples_locked(examples: list[dict]) -> None:
    """Записывает route_examples.jsonl. Вызывается ТОЛЬКО внутри _lock."""
    ROUTE_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False) for e in examples]
    ROUTE_EXAMPLES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_with_lock(fn: Callable[[list[dict]], list[dict]]) -> int:
    """Атомарный цикл: читает примеры под локом, применяет fn, записывает.

    fn получает список примеров, возвращает изменённый список.
    Возвращает разницу длин (>0 добавлено, <0 удалено, 0 без изменений).
    """
    with _lock:
        examples = _read_examples_locked()
        before = len(examples)
        examples = fn(examples)
        _write_examples_locked(examples)
        return len(examples) - before


# ─── публичные функции ───────────────────────────────────────────────────────

def run_learning_cycle() -> dict:
    """Full learning cycle. Returns stats dict."""
    from brain.feedback_store import load_all, archive_processed

    records = load_all()
    verified_success = [r for r in records if r["outcome"] == "success" and r["verified"]]
    verified_failure  = [r for r in records if r["outcome"] == "failure" and r["verified"]]
    auto_success      = [r for r in records if r["outcome"] == "success" and not r["verified"]]

    failure_texts = {r["text"].lower() for r in verified_failure}

    added_vs = 0
    added_auto = 0
    removed_n = 0

    def _apply(examples: list[dict]) -> list[dict]:
        nonlocal added_vs, added_auto, removed_n
        existing_texts = {e["text"].lower() for e in examples}

        # 1. Добавляем верифицированные успехи
        for rec in verified_success:
            text = rec["text"].strip()
            if text.lower() in existing_texts:
                continue
            examples.append({
                "text":      text,
                "route":     rec["route"],
                "tool":      rec["tool"],
                "tool_args": {},
                "filler":    "",
                "_source":   "learned_success",
                "_ts":       rec["ts"],
            })
            existing_texts.add(text.lower())
            added_vs += 1

        # 2. Удаляем примеры, которые вели к failure
        before_len = len(examples)
        examples = [
            e for e in examples
            if not (
                e["text"].lower() in failure_texts
                and e.get("_source") in ("auto_success", "learned_success")
            )
        ]
        removed_n = before_len - len(examples)

        # 3. Добавляем auto-success с высокой confidence
        for rec in auto_success:
            if rec.get("confidence", 0) < AUTO_SUCCESS_THRESHOLD:
                continue
            text = rec["text"].strip()
            if text.lower() in existing_texts:
                continue
            examples.append({
                "text":      text,
                "route":     rec["route"],
                "tool":      rec["tool"],
                "tool_args": {},
                "filler":    "",
                "_source":   "auto_success",
                "_ts":       rec["ts"],
            })
            existing_texts.add(text.lower())
            added_auto += 1

        return examples

    _run_with_lock(_apply)

    _invalidate_embed_cache()

    # Архивируем обработанные (feedback.jsonl остаётся маленьким)
    processed_ids = {r["id"] for r in verified_success + verified_failure}
    if processed_ids:
        archive_processed(processed_ids)

    # Читаем финальный счётчик без отдельного лока — достаточно точно для репорта
    try:
        with _lock:
            total = len(_read_examples_locked())
    except Exception:
        total = -1

    report = {
        "ts":               datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_feedback":   len(records),
        "verified_success": len(verified_success),
        "verified_failure":  len(verified_failure),
        "auto_success_added": added_auto,
        "examples_added":   added_vs + added_auto,
        "examples_removed": removed_n,
        "total_examples":   total,
    }
    _write_report(report)
    logger.info(
        f"[learning] Cycle done: +{report['examples_added']} added, "
        f"-{removed_n} removed, total={total}"
    )
    return report


def remove_failed_example(record_id: str) -> None:
    """
    Immediately removes auto-success example that was confirmed as failure.
    Called directly from explicit_feedback (no wait for nightly cycle).

    fix BUG-4: всё (load_all + поиск записи + удаление примера) выполняется
    ВНУТРИ одного _run_with_lock вызова. Это устраняет TOCTOU race condition:
    если archive_processed() удалил запись между load_all() и _run_with_lock(),
    в старой версии text_lower становился пустой строкой ("".lower() == ""),
    что приводило к удалению ВСЕХ auto_success/learned_success примеров.
    Теперь при не найденной записи _apply возвращает список без изменений.
    """
    from brain.feedback_store import load_all

    def _apply(examples: list[dict]) -> list[dict]:
        # Читаем feedback внутри лока, чтобы избежать TOCTOU
        records = load_all()
        rec = next((r for r in records if r["id"] == record_id), None)
        if not rec:
            # Запись уже архивирована или не существует — ничего не трогаем
            return examples
        text_lower = rec["text"].lower()
        if not text_lower:
            # Защита: никогда не удаляем примеры с пустым текстом
            logger.warning(f"[learning] remove_failed_example: пустой text для record_id={record_id!r}")
            return examples
        return [
            e for e in examples
            if not (
                e["text"].lower() == text_lower
                and e.get("_source") in ("auto_success", "learned_success")
            )
        ]

    diff = _run_with_lock(_apply)
    if diff < 0:
        _invalidate_embed_cache()
        logger.info(f"[learning] Removed failed example: record_id={record_id!r}")


# ─── legacy public API (для совместимости с router_embed.invalidate_cache) ──

def _load_examples() -> list[dict]:
    """Публичный read под локом. Используется внешним кодом (тесты, отладка)."""
    with _lock:
        return _read_examples_locked()


def _save_examples(examples: list[dict]) -> None:
    """Публичный write под локом. Используется внешним кодом (тесты)."""
    with _lock:
        _write_examples_locked(examples)


def _write_report(report: dict) -> None:
    try:
        LEARNING_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(LEARNING_REPORT, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[learning] Report write error: {e}")


def _invalidate_embed_cache() -> None:
    try:
        from brain.router_embed import invalidate_cache
        invalidate_cache()
    except Exception as e:
        logger.error(f"[learning] Cache invalidation error: {e}")


# Запуск напрямую: python -m brain.learning_loop
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    report = run_learning_cycle()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0)
