from __future__ import annotations

from dataclasses import dataclass
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


def _call_weather(location: str, language: str = "ru") -> Any:
    return get_weather(location=location, language=language)


def _call_crypto_search(query: str) -> Any:
    return search_coin(query=query)


def _call_crypto_price(ids: list[str], vs_currency: str = "usd") -> Any:
    return get_crypto_price(ids=ids, vs_currency=vs_currency)


def _call_currency_rates() -> Any:
    return get_rates()


def _call_currency_convert(amount: float, from_code: str, to_code: str) -> Any:
    return convert_currency(amount=amount, from_code=from_code, to_code=to_code)


def _call_time() -> Any:
    return get_time()


_TOOL_MAP: dict[str, Any] = {
    "weather": _call_weather,
    "crypto.search": _call_crypto_search,
    "crypto.price": _call_crypto_price,
    "currency.rates": _call_currency_rates,
    "currency.convert": _call_currency_convert,
    "time": _call_time,
}


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_TOOL_MAP.keys())


def call_tool(name: str, args: dict[str, Any] | None = None) -> ToolResult:
    args = args or {}
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return ToolResult(ok=False, error=f"Unknown tool: {name}. Available: {list_tools()}")
    try:
        result = fn(**args)
        return ToolResult(ok=True, data=result)
    except TypeError as e:
        return ToolResult(ok=False, error=f"Bad arguments for tool '{name}': {e}")
    except ValueError as e:
        return ToolResult(ok=False, error=str(e))
    except Exception as e:
        return ToolResult(ok=False, error=f"Tool '{name}' failed: {e}")
