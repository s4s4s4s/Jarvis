# brain/feedback_store.py
"""
Хранилище обратной связи для самообучения Jarvis.

Каждая запись:
{
  "id":         "uuid",
  "ts":         "2026-05-01T14:32:00Z",
  "text":       "запрос пользователя",
  "route":      "chat",
  "tool":       null,
  "confidence": 0.91,
  "source":     "embed",     # "embed" или "llm"
  "answer":     "ответ...",
  "outcome":    "success",   # "success" | "failure" | "unknown"
  "reason":     "",
  "verified":   false        # true = пользователь подтвердил вручную
}
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal

from core.paths import FEEDBACK_LOG

_lock = threading.Lock()

Outcome = Literal["success", "failure", "unknown"]


def record(
    *,
    text: str,
    route: str,
    tool: str | None,
    confidence: float,
    source: str,
    answer: str,
    outcome: Outcome,
    reason: str = "",
    verified: bool = False,
) -> str:
    """Records one feedback entry. Returns the record ID."""
    entry = {
        "id":         str(uuid.uuid4()),
        "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text":       text[:300],
        "route":      route,
        "tool":       tool,
        "confidence": round(confidence, 3),
        "source":     source,
        "answer":     answer[:500],
        "outcome":    outcome,
        "reason":     reason,
        "verified":   verified,
    }
    _write(entry)
    return entry["id"]


def mark_verified(record_id: str, correct: bool) -> None:
    """
    Marks a record as user-verified.
    Reads full file, updates target line, rewrites.
    """
    if not FEEDBACK_LOG.exists():
        return
    lines = FEEDBACK_LOG.read_text("utf-8").splitlines()
    updated = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("id") == record_id:
                obj["verified"] = True
                obj["outcome"] = "success" if correct else "failure"
            updated.append(json.dumps(obj, ensure_ascii=False))
        except json.JSONDecodeError:
            updated.append(line)
    with _lock:
        FEEDBACK_LOG.write_text("\n".join(updated) + "\n", encoding="utf-8")


def load_all() -> list[dict]:
    if not FEEDBACK_LOG.exists():
        return []
    result = []
    for line in FEEDBACK_LOG.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def archive_processed(record_ids: set[str]) -> None:
    """
    Moves processed records to feedback_archive.jsonl.
    Keeps feedback.jsonl small — only unprocessed records remain.
    """
    from core.paths import FEEDBACK_ARCHIVE
    if not FEEDBACK_LOG.exists():
        return
    lines = FEEDBACK_LOG.read_text("utf-8").splitlines()
    keep = []
    archive = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("id") in record_ids:
                archive.append(line)
            else:
                keep.append(line)
        except json.JSONDecodeError:
            keep.append(line)
    with _lock:
        FEEDBACK_LOG.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        if archive:
            with open(FEEDBACK_ARCHIVE, "a", encoding="utf-8") as f:
                f.write("\n".join(archive) + "\n")


def _write(entry: dict) -> None:
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _lock:
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"[feedback_store] Ошибка записи: {e}")
