# brain/router_embed.py
"""
Embedding-роутер на sentence-transformers.

Загружается один раз при старте, затем каждый запрос —
cosine similarity за ~20-50 ms. Возвращает None если
неуверенно → LLM fallback в ask.py.

fix #9: _load() защищён Lock + Double-Checked Locking.

fix N6: route_embed проверяет _example_vecs is not None перед @.

fix BUG-5: invalidate_cache() не выставляет None в начале — сборка
выполняется во временных переменных, затем атомарно присваивается пара.

fix C2: заменили отдельные _examples/_example_vecs на единый _state-кортеж.
Due to CPython GIL, присваивание одной ссылки (_state = ...) — одна инструкция
STORE_GLOBAL, а значит атомарно. Устраняет окно между двумя отдельными
присваиваниями, в котором route_embed мог видеть _examples=new/_example_vecs=old
(или наоборот) → IndexError или неверный маршрут.
route_embed читает state = _state один раз, затем работает с локальными копиями.

OPT-1: добавлен eager_load() — загрузка модели в фоновом потоке при старте Jarvis.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

import numpy as np

from core.config import EMBED_MODEL_NAME, EMBED_THRESHOLD, EMBED_AMBIGUITY
from core.paths import ROUTE_EXAMPLES

logger = logging.getLogger(__name__)

_model = None
# fix C2: единый _state-кортеж (examples, vecs) вместо двух отдельных переменных.
# Присваивание одной ссылки атомарно в CPython (STORE_GLOBAL = 1 байткод-инструкция).
_state: "tuple[list[dict], np.ndarray] | None" = None
_load_lock = threading.Lock()  # fix #9


def _load() -> None:
    global _model, _state
    if _model is not None:
        return
    with _load_lock:
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
        texts = [e["text"] for e in raw_examples]
        new_vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # fix C2: единое атомарное присваивание пары
        _state = (raw_examples, new_vecs)
        logger.info(f"[router_embed] Ready: {len(raw_examples)} examples loaded")


def eager_load() -> None:
    """OPT-1: загрузить SentenceTransformer заранее в фоновом потоке."""
    _load()


def route_embed(text: str) -> dict[str, Any] | None:
    """
    Returns dict with route/tool/tool_args/confidence/filler/_source
    or None if confidence is insufficient → LLM fallback needed.
    """
    if _model is None:
        _load()
    # fix C2: читаем _state один раз — гарантированно консистентная пара.
    state = _state
    if _model is None or state is None:
        return None
    examples, example_vecs = state

    try:
        vec = _model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        scores = (example_vecs @ vec.T).flatten()

        sorted_idx = np.argsort(scores)[::-1]
        top_idx   = int(sorted_idx[0])
        top_score = float(scores[top_idx])

        gap = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(sorted_idx) > 1 else 1.0

        if top_score < EMBED_THRESHOLD or gap < EMBED_AMBIGUITY:
            logger.debug(f"[router_embed] Uncertain: score={top_score:.3f} gap={gap:.3f} → LLM")
            return None

        best = examples[top_idx]
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
    """Force reload of examples and vectors (used by learning_loop).

    fix BUG-5 + C2: сборка new_vecs во временных переменных,
    затем одним атомарным присваиванием _state = (raw_examples, new_vecs).
    """
    global _state
    with _load_lock:
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
                if raw_examples:
                    texts = [e["text"] for e in raw_examples]
                    new_vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    _state = (raw_examples, new_vecs)  # fix C2: атомарно
                    logger.info(f"[router_embed] Cache refreshed: {len(raw_examples)} examples")
                else:
                    _state = None
                    logger.warning("[router_embed] route_examples.jsonl is empty after refresh")
            except Exception as e:
                logger.error(f"[router_embed] Cache refresh error: {e}")
                _state = None
        else:
            _state = None
