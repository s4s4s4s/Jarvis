from __future__ import annotations

import time
import ollama

from core.config import (
    OLLAMA_ROUTER_MODEL,
    OLLAMA_FAST_MODEL,
    OLLAMA_HEAVY_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_HEAVY_TIMEOUT,
    OLLAMA_RETRIES,
    OLLAMA_RETRY_DELAY,
)

MODEL_ROUTER = OLLAMA_ROUTER_MODEL
MODEL_FAST   = OLLAMA_FAST_MODEL
MODEL_HEAVY  = OLLAMA_HEAVY_MODEL

_OLLAMA_BASE_URL = "http://localhost:11434"

_client       = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
_client_heavy = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_HEAVY_TIMEOUT)


def is_ollama_available() -> bool:
    """Быстрая проверка доступности Ollama — вызывается при старте."""
    try:
        _client.list()
        return True
    except Exception:
        return False


def chat(model: str, messages: list[dict], options: dict | None = None) -> str:
    opts = options or {"temperature": 0.2, "num_ctx": 8192}
    client = _client_heavy if model == MODEL_HEAVY else _client
    last_err = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            resp = client.chat(model=model, messages=messages, options=opts)
            return resp.message.content.strip()
        except Exception as e:
            last_err = e
            print(f"[ollama] Ошибка (attempt {attempt + 1}): {e}")
            if attempt < OLLAMA_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
    print(f"[ollama] Все попытки исчерпаны: {last_err}")
    return ""
