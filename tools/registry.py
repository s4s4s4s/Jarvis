from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.crypto import get_crypto_price, search_coin
from tools.currency import convert_currency, get_rates
from tools.time_tool import get_time
from tools.weather import get_weather


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""


_TOOL_MAP = {
    "weather": (get_weather, ["location"]),
    "crypto.search": (search_coin, ["query"]),
    "crypto.price": (get_crypto_price, ["ids"]),
    "currency.rates": (lambda **_: get_rates(), []),
    "currency.convert": (convert_currency, ["amount", "from_code", "to_code"]),
    "time": (lambda **_: get_time(), []),
}


def call_tool(name: str, args: dict[str, Any] | None = None) -> ToolResult:
    args = args or {}

    handler_entry = _TOOL_MAP.get(name)
    if handler_entry is None:
        return ToolResult(ok=False, error=f"Unknown tool: {name}")

    fn, _ = handler_entry
    try:
        result = fn(**args)
        return ToolResult(ok=True, data=result)
    except TypeError as e:
        return ToolResult(ok=False, error=f"Bad arguments for tool '{name}': {e}")
    except ValueError as e:
        return ToolResult(ok=False, error=str(e))
    except Exception as e:
        return ToolResult(ok=False, error=f"Tool '{name}' failed: {e}")
