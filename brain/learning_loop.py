# brain/learning_loop.py
"""
Петля самообучения Jarvis.

Что делает:
1. Читает logs/feedback.jsonl
2. Верифицированные success → добавляет в route_examples.jsonl
3. Верифицированные failure → удаляет совпадающие auto-примеры
4. auto-success с confidence ≥ 0.90 → добавляет без верификации
5. Сбрасывает embed-кэш, архивирует обработанные записи
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

from core.paths import ROUTE_EXAMPLES, LEARNING_REPORT

logger = logging.getLogger(__name__)
_lock = threading.Lock()

AUTO_SUCCESS_THRESHOLD = 0.90  # confidence ниже этого → не добавляем


def run_learning_cycle() -> dict:
    """Full learning cycle. Returns stats dict."""
    from brain.feedback_store import load_all, archive_processed

    records = load_all()
    verified_success = [r for r in records if r["outcome"] == "success" and r["verified"]]
    verified_failure  = [r for r in records if r["outcome"] == "failure" and r["verified"]]
    auto_success      = [r for r in records if r["outcome"] == "success" and not r["verified"]]

    examples = _load_examples()
    existing_texts = {e["text"].lower() for e in examples}

    added   = 0
    removed = 0

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
        added += 1

    # 2. Удаляем примеры, которые вели к failure
    failure_texts = {r["text"].lower() for r in verified_failure}
    before_len = len(examples)
    examples = [
        e for e in examples
        if not (
            e["text"].lower() in failure_texts
            and e.get("_source") in ("auto_success", "learned_success")
        )
    ]
    removed = before_len - len(examples)

    # 3. Добавляем auto-success с высокой confidence
    auto_added = 0
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
        auto_added += 1

    _save_examples(examples)
    _invalidate_embed_cache()

    # Архивируем обработанные (feedback.jsonl остаётся маленьким)
    processed_ids = {r["id"] for r in verified_success + verified_failure}
    if processed_ids:
        archive_processed(processed_ids)

    report = {
        "ts":               datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_feedback":   len(records),
        "verified_success": len(verified_success),
        "verified_failure":  len(verified_failure),
        "auto_success_added": auto_added,
        "examples_added":   added + auto_added,
        "examples_removed": removed,
        "total_examples":   len(examples),
    }
    _write_report(report)
    logger.info(
        f"[learning] Cycle done: +{report['examples_added']} added, "
        f"-{removed} removed, total={len(examples)}"
    )
    return report


def remove_failed_example(record_id: str) -> None:
    """
    Immediately removes auto-success example that was confirmed as failure.
    Called directly from explicit_feedback (no wait for nightly cycle).
    """
    from brain.feedback_store import load_all
    records = load_all()
    rec = next((r for r in records if r["id"] == record_id), None)
    if not rec:
        return
    examples = _load_examples()
    text_lower = rec["text"].lower()
    before = len(examples)
    examples = [
        e for e in examples
        if not (
            e["text"].lower() == text_lower
            and e.get("_source") in ("auto_success", "learned_success")
        )
    ]
    if len(examples) < before:
        _save_examples(examples)
        _invalidate_embed_cache()
        logger.info(f"[learning] Removed failed example: '{rec['text'][:60]}'")


def _load_examples() -> list[dict]:
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


def _save_examples(examples: list[dict]) -> None:
    with _lock:
        ROUTE_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(e, ensure_ascii=False) for e in examples]
        ROUTE_EXAMPLES.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
