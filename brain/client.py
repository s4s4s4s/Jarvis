from __future__ import annotations

import time
import ollama

from core.config import (
    OLLAMA_ROUTER_MODEL,
    OLLAMA_FAST_MODEL,
    OLLAMA_HEAVY_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_RETRIES,
    OLLAMA_RETRY_DELAY,
)

# Re-export for consumers that import MODEL_* from brain.client
MODEL_ROUTER = OLLAMA_ROUTER_MODEL
MODEL_FAST   = OLLAMA_FAST_MODEL
MODEL_HEAVY  = OLLAMA_HEAVY_MODEL

_OLLAMA_BASE_URL = "http://localhost:11434"
_client = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)


def chat(model: str, messages: list[dict], options: dict | None = None) -> str:
    opts = options or {"temperature": 0.2, "num_ctx": 8192}
    last_err = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            resp = _client.chat(model=model, messages=messages, options=opts)
            return resp.message.content.strip()
        except Exception as e:
            last_err = e
            print(f"[ollama] Ошибка (попытка {attempt + 1}): {e}")
            if attempt < OLLAMA_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
    print(f"[ollama] Все попытки исчерпаны: {last_err}")
    return ""
