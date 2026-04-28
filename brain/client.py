"""brain/client.py

Dual-backend LLM client:
  - Ollama  (default, serial)          → http://localhost:11434
  - llama-server (parallel, optional)  → http://127.0.0.1:8080  (OpenAI-compat API)

Backend is selected per call via the `backend` parameter or the global
`set_backend()` helper.  The Executor switches to 'llama' before running
parallel tasks and switches back to 'ollama' afterwards.
"""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Literal

import httpx
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

logger = logging.getLogger(__name__)

MODEL_ROUTER = OLLAMA_ROUTER_MODEL
MODEL_FAST   = OLLAMA_FAST_MODEL
MODEL_HEAVY  = OLLAMA_HEAVY_MODEL

# ---------------------------------------------------------------------------
# Ollama clients (unchanged)
# ---------------------------------------------------------------------------
_OLLAMA_BASE_URL = "http://localhost:11434"
_client       = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
_client_heavy = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_HEAVY_TIMEOUT)

# ---------------------------------------------------------------------------
# llama-server settings (OpenAI-compatible /v1/chat/completions)
# ---------------------------------------------------------------------------
_LLAMA_BASE_URL  = "http://127.0.0.1:8080"
_LLAMA_TIMEOUT   = 300  # seconds — generation can be long for 32B

# ---------------------------------------------------------------------------
# Global backend selector
# ---------------------------------------------------------------------------
BackendType = Literal["ollama", "llama"]
_active_backend: BackendType = "ollama"


def set_backend(backend: BackendType) -> None:
    """Switch the global backend used by chat()."""
    global _active_backend
    _active_backend = backend
    logger.info("[client] Backend switched to '%s'", backend)


def get_backend() -> BackendType:
    return _active_backend


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def is_ollama_available() -> bool:
    """Quick check — called at startup."""
    try:
        _client.list()
        return True
    except Exception:
        return False


def is_llama_server_available(base_url: str = _LLAMA_BASE_URL) -> bool:
    try:
        with httpx.Client(timeout=4) as client:
            resp = client.get(f"{base_url}/health")
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Synchronous chat  (Ollama backend)
# ---------------------------------------------------------------------------

def _chat_ollama(model: str, messages: list[dict], options: dict | None = None) -> str:
    opts = options or {"temperature": 0.2, "num_ctx": 8192}
    client = _client_heavy if model == MODEL_HEAVY else _client
    last_err = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            resp = client.chat(model=model, messages=messages, options=opts)
            return resp.message.content.strip()
        except Exception as e:
            last_err = e
            logger.warning("[ollama] Error (attempt %d): %s", attempt + 1, e)
            if attempt < OLLAMA_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
    logger.error("[ollama] All retries exhausted: %s", last_err)
    return ""


# ---------------------------------------------------------------------------
# Async chat  (llama-server backend — OpenAI /v1/chat/completions)
# ---------------------------------------------------------------------------

async def _chat_llama_async(
    messages: list[dict],
    base_url: str = _LLAMA_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=_LLAMA_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def _chat_llama_sync(
    messages: list[dict],
    base_url: str = _LLAMA_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """Sync wrapper around the async llama call (for non-async callers)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop — use a thread-safe approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    _chat_llama_async(messages, base_url, temperature, max_tokens),
                )
                return future.result(timeout=_LLAMA_TIMEOUT)
        return loop.run_until_complete(
            _chat_llama_async(messages, base_url, temperature, max_tokens)
        )
    except Exception as e:
        logger.error("[llama-server] chat error: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def chat(
    model: str,
    messages: list[dict],
    options: dict | None = None,
    backend: BackendType | None = None,
) -> str:
    """
    Send a chat request.
    - If backend='llama' (or global backend is 'llama'), use llama-server.
    - Otherwise use Ollama.
    `model` is ignored for llama-server (model is fixed at server start).
    """
    effective = backend or _active_backend
    if effective == "llama":
        temperature = (options or {}).get("temperature", 0.2)
        return _chat_llama_sync(messages, temperature=temperature)
    return _chat_ollama(model, messages, options)


async def chat_async(
    model: str,
    messages: list[dict],
    options: dict | None = None,
    backend: BackendType | None = None,
) -> str:
    """
    Async version — used by the parallel executor path.
    """
    effective = backend or _active_backend
    if effective == "llama":
        temperature = (options or {}).get("temperature", 0.2)
        return await _chat_llama_async(messages, temperature=temperature)
    # Ollama is blocking — run in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: _chat_ollama(model, messages, options)
    )


# ---------------------------------------------------------------------------
# Model warmup (eliminates cold-start latency)
# ---------------------------------------------------------------------------

def _warmup_model(model: str, use_heavy_client: bool = False) -> None:
    client = _client_heavy if use_heavy_client else _client
    try:
        logger.info("[warmup] Pinging model '%s'...", model)
        client.chat(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "temperature": 0.0},
            keep_alive="1h",
        )
        logger.info("[warmup] Model '%s' loaded into VRAM v", model)
    except Exception as e:
        logger.warning("[warmup] Model '%s' warmup failed: %s", model, e)


def warmup_all(blocking: bool = False) -> None:
    import threading
    targets = [
        (MODEL_ROUTER, False),
        (MODEL_FAST,   False),
        (MODEL_HEAVY,  True),
    ]
    threads = [
        threading.Thread(target=_warmup_model, args=(m, h), daemon=True)
        for m, h in targets
    ]
    for t in threads:
        t.start()
    if blocking:
        for t in threads:
            t.join()


warmup_all(blocking=False)
