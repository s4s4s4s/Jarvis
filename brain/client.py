from __future__ import annotations

import logging
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
    PROJECT_CODER_MODEL,
    PROJECT_REVIEWER_MODEL,
    PROJECT_ARCHITECT_MODEL,
    PROJECT_HEALER_MODEL,
    PROJECT_INTAKE_MODEL,
    PROJECT_README_MODEL,
    PROJECT_REPORT_MODEL,
)

MODEL_ROUTER = OLLAMA_ROUTER_MODEL
MODEL_FAST   = OLLAMA_FAST_MODEL
MODEL_HEAVY  = OLLAMA_HEAVY_MODEL

# P4: ролевые алиасы — из config.py, с понятными именами для оркестратора.
MODEL_CODER     = PROJECT_CODER_MODEL
MODEL_REVIEWER  = PROJECT_REVIEWER_MODEL
MODEL_ARCHITECT = PROJECT_ARCHITECT_MODEL
MODEL_HEALER    = PROJECT_HEALER_MODEL
MODEL_INTAKE    = PROJECT_INTAKE_MODEL
MODEL_README    = PROJECT_README_MODEL
MODEL_REPORT    = PROJECT_REPORT_MODEL

_OLLAMA_BASE_URL = "http://localhost:11434"

_client       = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
_client_heavy = ollama.Client(host=_OLLAMA_BASE_URL, timeout=OLLAMA_HEAVY_TIMEOUT)

# fix M2: используем logger вместо print() — ошибки попадают в structured log
# и становятся видны auditor.py / nightly_self_heal.py.
logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """raised when chat() не получил осмысленный ответ после всех retry."""


def is_ollama_available() -> bool:
    """Быстрая проверка доступности Ollama — вызывается при старте."""
    try:
        _client.list()
        return True
    except Exception:
        return False


def chat(model: str, messages: list[dict], options: dict | None = None) -> str:
    """Отправить запрос в ollama. Возвращает текст или бросает LLMError.

    fix #1: перестал возвращать "" — вместо этого бросает LLMError, чтобы
    вызывающие код (project.py, planner.py, self_extend.py) могли войти в except
    и вернуть понятный fallback вместо создания пустых файлов.

    fix #10: MODEL_INTAKE, MODEL_README, MODEL_REPORT добавлены в heavy_models,
    чтобы получать OLLAMA_HEAVY_TIMEOUT вместо короткого OLLAMA_TIMEOUT.
    Эти модели используются в длинных фазах _intake/_architect/_healer/_readme/_report.

    fix M2: заменили print() на logger.warning/error — ошибки теперь попадают
    в structured log и видны auditor.py / nightly_self_heal.py.
    """
    opts = options or {"temperature": 0.2, "num_ctx": 8192}
    heavy_models = {
        MODEL_HEAVY,
        MODEL_CODER, MODEL_REVIEWER, MODEL_ARCHITECT, MODEL_HEALER,
        MODEL_INTAKE, MODEL_README, MODEL_REPORT,
    }
    client = _client_heavy if model in heavy_models else _client
    last_err: Exception | None = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            resp = client.chat(model=model, messages=messages, options=opts)
            content = resp.message.content
            if content:
                return content.strip()
            last_err = LLMError(f"empty response from model '{model}'")
            logger.warning("[ollama] Пустой ответ (attempt %d), retry...", attempt + 1)
        except Exception as e:
            last_err = e
            logger.warning("[ollama] Ошибка (attempt %d): %s", attempt + 1, e)
        if attempt < OLLAMA_RETRIES:
            time.sleep(OLLAMA_RETRY_DELAY)
    logger.error("[ollama] Все попытки исчерпаны: %s", last_err)
    raise LLMError(f"ollama '{model}' failed after {OLLAMA_RETRIES + 1} attempts: {last_err}")
