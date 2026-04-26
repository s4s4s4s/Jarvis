from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import requests

_COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
_COINGECKO_SEARCH_URL  = "https://api.coingecko.com/api/v3/search"
_TTL_PRICE  = 60    # seconds — public API rate limit protection
_TTL_SEARCH = 300   # search results change rarely

# FIX: retry-параметры для обработки 429 Too Many Requests от CoinGecko
_MAX_RETRIES   = 3
_RETRY_DELAYS  = [2.0, 5.0, 10.0]  # экспоненциальный back-off

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


def _get_with_retry(url: str, params: dict, timeout: int = 15) -> requests.Response:
    """
    GET-запрос с retry при 429 (rate limit) от CoinGecko public API.
    Бросает requests.HTTPError при финальной неудаче.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"accept": "application/json"},
            )
            if resp.status_code == 429:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[crypto] 429 rate limit, жду {delay}s (попытка {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(delay)
                last_exc = requests.HTTPError(f"429 Too Many Requests", response=resp)
                continue
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[crypto] 429 rate limit, жду {delay}s")
                time.sleep(delay)
                last_exc = e
                continue
            raise
        except Exception as e:
            raise
    # Все попытки исчерпаны
    raise ValueError(
        "Лимит запросов к CoinGecko исчерпан. Попробуйте через минуту."
    ) from last_exc


def search_coin(query: str) -> list[dict[str, Any]]:
    key = f"search:{query.lower().strip()}"
    cached = _get(key)
    if cached is not None:
        return cached

    response = _get_with_retry(_COINGECKO_SEARCH_URL, params={"query": query})
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

    response = _get_with_retry(
        _COINGECKO_MARKETS_URL,
        params={
            "vs_currency": vs_currency,
            "ids": ",".join(ids),
            "price_change_percentage": "24h",
        },
    )
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
