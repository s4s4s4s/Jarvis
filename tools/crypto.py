from __future__ import annotations

from typing import Any, Dict, List

import requests

_COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
_COINGECKO_SEARCH_URL = "https://api.coingecko.com/api/v3/search"


def search_coin(query: str) -> list[dict[str, Any]]:
    response = requests.get(
        _COINGECKO_SEARCH_URL,
        params={"query": query},
        timeout=15,
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    coins: List[Dict[str, Any]] = data.get("coins") or []
    return [
        {
            "id": coin.get("id"),
            "symbol": coin.get("symbol"),
            "name": coin.get("name"),
            "market_cap_rank": coin.get("market_cap_rank"),
        }
        for coin in coins[:10]
    ]


def get_crypto_price(ids: list[str], vs_currency: str = "usd") -> list[dict[str, Any]]:
    response = requests.get(
        _COINGECKO_MARKETS_URL,
        params={
            "vs_currency": vs_currency,
            "ids": ",".join(ids),
            "price_change_percentage": "24h",
        },
        timeout=15,
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    return [
        {
            "id": item.get("id"),
            "symbol": item.get("symbol"),
            "name": item.get("name"),
            "current_price": item.get("current_price"),
            "market_cap": item.get("market_cap"),
            "market_cap_rank": item.get("market_cap_rank"),
            "price_change_percentage_24h": item.get("price_change_percentage_24h"),
        }
        for item in data
    ]
