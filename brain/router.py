from __future__ import annotations

import json
from typing import Any

from brain.prompts import ROUTER_SYSTEM


def parse_router_response(text: str) -> dict[str, Any]:
    data = json.loads(text)
    return {
        "route": data.get("route", "chat"),
        "tool": data.get("tool"),
        "tool_args": data.get("tool_args") or {},
        "confidence": data.get("confidence", 0),
        "filler": data.get("filler", ""),
        "reason": data.get("reason", ""),
    }
