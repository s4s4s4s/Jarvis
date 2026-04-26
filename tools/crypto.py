from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import requests

_COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
_COINGECKO_SEARCH_URL = "https://api.coingecko.com/api/v3/search"
_TTL_PRICE = 60    # seconds — public API rate limit protection
_TTL_SEARCH = 300  # search results change rarely

_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}  # key -> (expires_at, value)


def _get(key: str) -> Any | None:
    with _cache_lock:
        item = _cache.get(key)
    if item is None:
        return None
    expires_at, value = item
    if time.time() > expires_at:
        with _cache_lock:
            _cache.pop(key, None)
        return None
    return value


def _set(key: str, value: Any, ttl: int) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + ttl, value)


def search_coin(query: str) -> list[dict[str, Any]]:
    key = f"search:{query.lower().strip()}"
    cached = _get(key)
    if cached is not None:
        return cached

    response = requests.get(
        _COINGECKO_SEARCH_URL,
        params={"query": query},
        timeout=15,
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    coins: List[Dict[str, Any]] = data.get("coins") or []
    result = [
        {
            "id": coin.get("id"),
            "symbol": coin.get("symbol"),
            "name": coin.get("name"),
            "market_cap_rank": coin.get("market_cap_rank"),
        }
        for coin in coins[:10]
    ]
    _set(key, result, _TTL_SEARCH)
    return result


def get_crypto_price(ids: list[str], vs_currency: str = "usd") -> list[dict[str, Any]]:
    key = f"price:{','.join(sorted(ids))}:{vs_currency}"
    cached = _get(key)
    if cached is not None:
        return cached

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
    result = [
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
    _set(key, result, _TTL_PRICE)
    return result
