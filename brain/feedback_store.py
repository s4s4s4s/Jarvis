# brain/feedback_store.py
"""
Хранилище обратной связи для самообучения Jarvis.

Каждая запись в logs/feedback.jsonl:
{
  "id":         str (uuid),
  "ts":         str (ISO 8601),
  "text":       str (запрос пользователя),
  "route":      str,
  "tool":       str | null,
  "confidence": float,
  "source":     "embed" | "llm",
  "answer":     str (первые 500 символов ответа),
  "outcome":    "success" | "failure" | "unknown",
  "reason":     str,
  "verified":   bool (true = пользователь подтвердил вручную)
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
    """Записывает оценку одного ответа. Возвращает ID записи."""
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
    """Помечает запись как верифицированную пользователем."""
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
    """Загружает все записи из feedback.jsonl."""
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


def _write(entry: dict) -> None:
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _lock:
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"[feedback_store] Ошибка записи: {e}")
