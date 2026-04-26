from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import requests

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TTL_SECONDS = 600
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _cache_get(key: tuple[str, str]) -> Optional[dict[str, Any]]:
    with _cache_lock:
        item = _cache.get(key)
    if not item:
        return None
    expires_at, value = item
    if time.time() > expires_at:
        with _cache_lock:
            _cache.pop(key, None)
        return None
    return value


def _cache_set(key: tuple[str, str], value: dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + _TTL_SECONDS, value)


def geocode_location(query: str, language: str = "ru") -> dict[str, Any]:
    key = ("geocode", f"{query}:{language}")
    cached = _cache_get(key)
    if cached is not None:
        return cached

    response = requests.get(
        _GEOCODE_URL,
        params={"name": query, "count": 1, "language": language, "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    results: List[Dict[str, Any]] = data.get("results") or []
    if not results:
        raise ValueError(f"Location not found: {query}")

    result = {
        "name": results[0].get("name"),
        "country": results[0].get("country"),
        "admin1": results[0].get("admin1"),
        "latitude": results[0].get("latitude"),
        "longitude": results[0].get("longitude"),
        "timezone": results[0].get("timezone"),
    }
    _cache_set(key, result)
    return result


def get_weather(location: str, language: str = "ru") -> dict[str, Any]:
    geo = geocode_location(location, language=language)
    key = ("weather", f"{geo['latitude']}:{geo['longitude']}:{language}")
    cached = _cache_get(key)
    if cached is not None:
        return cached

    response = requests.get(
        _FORECAST_URL,
        params={
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
            ],
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    result = {
        "location": geo,
        "current": payload.get("current", {}),
        "daily": payload.get("daily", {}),
    }
    _cache_set(key, result)
    return result
