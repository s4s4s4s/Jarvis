# brain/router_embed.py
"""
Embedding-роутер на sentence-transformers.

Zagrużaetsja odin raz pri starte, zathem každyj zapros —
cosine similarity za ~20-50 ms. Vozvrashhaet None esli
neuvereno → LLM fallback v ask.py.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from core.config import EMBED_MODEL_NAME, EMBED_THRESHOLD, EMBED_AMBIGUITY
from core.paths import ROUTE_EXAMPLES

logger = logging.getLogger(__name__)

_model = None
_examples: list[dict] | None = None
_example_vecs: np.ndarray | None = None


def _load() -> None:
    global _model, _examples, _example_vecs
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("[router_embed] sentence-transformers not installed. Run: pip install sentence-transformers>=2.7.0")
        return

    if not ROUTE_EXAMPLES.exists():
        logger.warning(f"[router_embed] {ROUTE_EXAMPLES} not found — embed router disabled")
        return

    raw_examples = []
    for line in ROUTE_EXAMPLES.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_examples.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not raw_examples:
        logger.warning("[router_embed] route_examples.jsonl is empty")
        return

    logger.info(f"[router_embed] Loading model: {EMBED_MODEL_NAME}")
    _model = SentenceTransformer(EMBED_MODEL_NAME)
    _examples = raw_examples
    texts = [e["text"] for e in _examples]
    _example_vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    logger.info(f"[router_embed] Ready: {len(_examples)} examples loaded")


def route_embed(text: str) -> dict[str, Any] | None:
    """
    Returns dict with route/tool/tool_args/confidence/filler/_source
    or None if confidence is insufficient → LLM fallback needed.
    """
    if _model is None:
        _load()
    if _model is None or _examples is None or _example_vecs is None:
        return None

    try:
        vec = _model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        scores = (_example_vecs @ vec.T).flatten()

        sorted_idx = np.argsort(scores)[::-1]
        top_idx   = int(sorted_idx[0])
        top_score = float(scores[top_idx])

        gap = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(sorted_idx) > 1 else 1.0

        if top_score < EMBED_THRESHOLD or gap < EMBED_AMBIGUITY:
            logger.debug(f"[router_embed] Uncertain: score={top_score:.3f} gap={gap:.3f} → LLM")
            return None

        best = _examples[top_idx]
        logger.debug(f"[router_embed] Hit: '{best['text']}' score={top_score:.3f} gap={gap:.3f}")
        return {
            "route":     best["route"],
            "tool":      best.get("tool"),
            "tool_args": best.get("tool_args") or {},
            "confidence": round(top_score, 3),
            "filler":    best.get("filler") or "",
            "reason":    f"embed match='{best['text'][:40]}' score={top_score:.3f}",
            "_source":   "embed",
        }
    except Exception as e:
        logger.error(f"[router_embed] Error: {e}")
        return None


def invalidate_cache() -> None:
    """Force reload of examples and vectors on next call (used by learning_loop)."""
    global _model, _examples, _example_vecs
    _examples = None
    _example_vecs = None
    # Keep _model loaded — only reload examples/vectors, not the heavy model
    if _model is not None and ROUTE_EXAMPLES.exists():
        try:
            raw_examples = []
            for line in ROUTE_EXAMPLES.read_text("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            _examples = raw_examples
            texts = [e["text"] for e in _examples]
            _example_vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            logger.info(f"[router_embed] Cache refreshed: {len(_examples)} examples")
        except Exception as e:
            logger.error(f"[router_embed] Cache refresh error: {e}")
            _examples = None
            _example_vecs = None
