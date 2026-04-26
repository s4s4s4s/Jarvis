from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import ROUTER_LOG_PATH

_lock = threading.Lock()


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


def _write(entry: dict[str, Any]) -> None:
    try:
        p = Path(ROUTER_LOG_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"[logger] \u041e\u0448\u0438\u0431\u043a\u0430 \u0437\u0430\u043f\u0438\u0441\u0438: {e}")
