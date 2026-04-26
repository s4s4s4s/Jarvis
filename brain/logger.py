from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ROUTER_LOG

_lock = threading.Lock()
_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def log_route(
    text: str,
    route: str,
    tool: str | None,
    confidence: float,
    reason: str,
    answer_ms: int,
) -> None:
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": text[:200],
        "route": route,
        "tool": tool,
        "confidence": round(confidence, 3),
        "reason": reason,
        "answer_ms": answer_ms,
    }
    _write(entry)


def _rotate_if_needed(p: Path) -> None:
    """Rename router.jsonl -> router.jsonl.1 when file exceeds _MAX_SIZE_BYTES."""
    try:
        if p.exists() and os.path.getsize(p) >= _MAX_SIZE_BYTES:
            rotated = Path(str(p) + ".1")
            if rotated.exists():
                rotated.unlink()
            p.rename(rotated)
    except Exception as e:
        print(f"[logger] Rotation error: {e}")


def _write(entry: dict[str, Any]) -> None:
    try:
        p = ROUTER_LOG  # Path object from core.paths — directory already created
        line = json.dumps(entry, ensure_ascii=False)
        with _lock:
            _rotate_if_needed(p)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"[logger] Ошибка записи: {e}")
