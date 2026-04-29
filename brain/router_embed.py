# brain/router_embed.py
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from core.config import EMBED_THRESHOLD, EMBED_AMBIGUITY, EMBED_MODEL_NAME
from core.paths import ROUTE_EXAMPLES

logger = logging.getLogger(__name__)

_model = None
_examples: list[dict] | None = None
_example_vecs: "np.ndarray | None" = None


def _load() -> None:
    global _model, _examples, _example_vecs
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        _examples = [
            json.loads(line)
            for line in ROUTE_EXAMPLES.read_text("utf-8").splitlines()
            if line.strip()
        ]
        texts = [e["text"] for e in _examples]
        _example_vecs = _model.encode(texts, normalize_embeddings=True)
        logger.info(f"[router_embed] Загружено {len(_examples)} примеров")
    except Exception as e:
        logger.error(f"[router_embed] Ошибка загрузки: {e}")
        _model = None
        _examples = None
        _example_vecs = None


def route_embed(text: str) -> dict[str, Any] | None:
    """
    Быстрый embedding-роутер.
    Возвращает dict с route/tool/tool_args/confidence/_source
    или None если уверенности недостаточно — нужен LLM fallback.
    """
    _load()
    if _model is None or _examples is None or _example_vecs is None:
        return None

    try:
        vec = _model.encode([text], normalize_embeddings=True)
        scores = (_example_vecs @ vec.T).flatten()
        sorted_idx = np.argsort(scores)[::-1]
        top_idx = int(sorted_idx[0])
        top_score = float(scores[top_idx])

        gap = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(scores) > 1 else 1.0

        if top_score < EMBED_THRESHOLD or gap < EMBED_AMBIGUITY:
            return None

        best = _examples[top_idx]
        return {
            "route":      best["route"],
            "tool":       best.get("tool"),
            "tool_args":  best.get("tool_args") or {},
            "confidence": round(top_score, 3),
            "filler":     best.get("filler", ""),
            "reason":     f"embed similarity={top_score:.3f}",
            "_source":    "embed",
        }
    except Exception as e:
        logger.error(f"[router_embed] Ошибка инференса: {e}")
        return None


def invalidate_cache() -> None:
    """Сбросить кэш — вызывается learning_loop после обновления примеров."""
    global _model, _examples, _example_vecs
    _model = None
    _examples = None
    _example_vecs = None
    logger.info("[router_embed] Кэш сброшен")
