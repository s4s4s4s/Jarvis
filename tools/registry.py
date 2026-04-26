from __future__ import annotations

from typing import Any

from tools.crypto import get_crypto_price, search_coin
from tools.currency import convert_currency, get_rates
from tools.time_tool import get_time
from tools.weather import get_weather


def call_tool(name: str, args: dict[str, Any] | None = None) -> Any:
    args = args or {}

    if name == "weather":
        return get_weather(**args)
    if name == "crypto.search":
        return search_coin(**args)
    if name == "crypto.price":
        return get_crypto_price(**args)
    if name == "currency.rates":
        return get_rates()
    if name == "currency.convert":
        return convert_currency(**args)
    if name == "time":
        return get_time()

    raise ValueError(f"Unknown tool: {name}")
