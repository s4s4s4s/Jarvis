from __future__ import annotations

import json
from typing import Any

from brain.prompts import ROUTER_SYSTEM  # noqa: F401 — re-exported for convenience


def parse_router_response(text: str) -> dict[str, Any]:
    """Parse router JSON response with graceful fallback on invalid JSON."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "route": "chat",
            "tool": None,
            "tool_args": {},
            "confidence": 0.0,
            "filler": "",
            "reason": "json_parse_error",
        }
    return {
        "route": data.get("route", "chat"),
        "tool": data.get("tool"),
        "tool_args": data.get("tool_args") or {},
        "confidence": data.get("confidence", 0.0),
        "filler": data.get("filler") or "",
        "reason": data.get("reason") or "",
    }
