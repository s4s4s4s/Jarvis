# brain/learning_loop.py
"""
Петля самообучения Jarvis.

Что делает:
1. Читает logs/feedback.jsonl
2. Верифицированные success → добавляет в data/route_examples.jsonl
3. Верифицированные failure → удаляет похожие авто-примеры из базы
4. Auto-success с confidence >= 0.90 → добавляет как новые примеры
5. Сбрасывает embedding-кэш
6. Пишет отчёт в logs/learning_report.jsonl
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

from brain.feedback_store import load_all
from core.paths import ROUTE_EXAMPLES, LEARNING_REPORT

logger = logging.getLogger(__name__)
_lock = threading.Lock()

AUTO_SUCCESS_MIN_CONFIDENCE = 0.90


def run_learning_cycle() -> dict:
    """Полный цикл обучения. Возвращает статистику."""
    records = load_all()

    verified_success = [r for r in records if r["outcome"] == "success" and r["verified"]]
    verified_failure = [r for r in records if r["outcome"] == "failure" and r["verified"]]
    auto_success     = [r for r in records if r["outcome"] == "success" and not r["verified"]]

    examples = _load_examples()
    existing_texts = {e["text"].lower() for e in examples}

    added = 0
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

    # 2. Удаляем примеры, которые привели к failure
    failure_texts = {r["text"].lower() for r in verified_failure}
    before_len = len(examples)
    examples = [
        e for e in examples
        if not (
            e["text"].lower() in failure_texts
            and e.get("_source") in ("learned_success", "auto_success")
        )
    ]
    removed = before_len - len(examples)

    # 3. Добавляем auto_success с высокой confidence
    for rec in auto_success:
        if rec.get("confidence", 0) < AUTO_SUCCESS_MIN_CONFIDENCE:
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
        added += 1

    _save_examples(examples)
    _invalidate_embed_cache()

    report = {
        "ts":                datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_feedback":    len(records),
        "verified_success":  len(verified_success),
        "verified_failure":  len(verified_failure),
        "auto_success_used": sum(1 for r in auto_success if r.get("confidence", 0) >= AUTO_SUCCESS_MIN_CONFIDENCE),
        "examples_added":    added,
        "examples_removed":  removed,
        "total_examples":    len(examples),
    }
    _write_report(report)
    logger.info(
        f"[learning] Цикл завершён: +{added} примеров, -{removed} удалено. "
        f"Всего в базе: {len(examples)}"
    )
    return report


def add_failure_example(record_id: str) -> None:
    """Немедленно удаляет авто-пример после подтверждённой ошибки."""
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
            and e.get("_source") in ("learned_success", "auto_success")
        )
    ]
    if len(examples) < before:
        _save_examples(examples)
        _invalidate_embed_cache()
        logger.info(f"[learning] Удалён авто-пример: {rec['text'][:60]}")


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
        logger.error(f"[learning] Ошибка записи отчёта: {e}")


def _invalidate_embed_cache() -> None:
    try:
        import brain.router_embed as re_mod
        re_mod.invalidate_cache()
    except Exception as e:
        logger.error(f"[learning] Ошибка сброса embed-кэша: {e}")


if __name__ == "__main__":
    report = run_learning_cycle()
    print(json.dumps(report, ensure_ascii=False, indent=2))
