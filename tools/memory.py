"""
tools/memory.py — долгосрочная векторная память Jarvis (Level 2).

Бэкенд: ChromaDB (локальный, персистентный) +
         sentence-transformers/all-MiniLM-L6-v2 (локально, 90 MB, быстро).

Публичный API (полностью обратно совместим с предыдущей версией):
  add_fact(fact, category, source) → bool
  get_memory_context(max_facts)    → str
  search_memory(query, n_results)  → list[dict]
  get_all_facts()                  → list[dict]
  extract_and_save_async(user, bot)

Миграция: если data/memory.json существует и векторная БД пуста —
         автоматически импортируем все факты при первом запуске.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from datetime import datetime
from typing import Any

from core.config import (
    OLLAMA_FAST_MODEL,
    MEMORY_MAX_FACTS,
    MEMORY_CONTEXT_FACTS,
    MEMORY_SEARCH_FACTS,
    MEMORY_SIM_THRESHOLD,
    MEMORY_EMBED_MODEL,
    MEMORY_COLLECTION,
)
from core.paths import MEMORY_PATH, CHROMA_DIR
from brain.client import chat

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Инициализация ChromaDB и SentenceTransformer
# ───────────────────────────────────────────────────────────────────────────────

_init_lock   = threading.Lock()
_embed_model = None   # SentenceTransformer (lazy load)
_collection  = None   # chromadb.Collection (lazy load)


def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    with _init_lock:
        if _embed_model is not None:
            return _embed_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[memory] Загрузка embed-модели {MEMORY_EMBED_MODEL}...")
            _embed_model = SentenceTransformer(MEMORY_EMBED_MODEL)
            logger.info("[memory] embed-модель готова")
        except ImportError:
            logger.error("[memory] sentence-transformers не установлен! pip install sentence-transformers")
            raise
    return _embed_model


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    with _init_lock:
        if _collection is not None:
            return _collection
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("[memory] chromadb не установлен! pip install chromadb")
            raise

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[memory] ChromaDB коллекция '{MEMORY_COLLECTION}' готова, "
                    f"{_collection.count()} записей")

        # Автомиграция из legacy flat-JSON
        _maybe_migrate()
    return _collection


# ───────────────────────────────────────────────────────────────────────────────
# Миграция из flat JSON → ChromaDB
# ───────────────────────────────────────────────────────────────────────────────

def _maybe_migrate() -> None:
    """Если data/memory.json существует и коллекция пуста — импортируем."""
    col = _collection
    if col is None or col.count() > 0:
        return
    if not MEMORY_PATH.exists():
        return
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            facts = json.load(f)
        if not isinstance(facts, list) or not facts:
            return
        logger.info(f"[memory] Миграция {len(facts)} фактов из memory.json → ChromaDB...")
        embed = _get_embed_model()
        texts = [e.get("fact", "") for e in facts]
        ids   = [e.get("id") or str(uuid.uuid4()) for e in facts]
        metas = [
            {
                "category":   e.get("category", "общее"),
                "created_at": e.get("created_at", ""),
                "source":     (e.get("source") or "")[:200],
            }
            for e in facts
        ]
        embeddings = embed.encode(texts, normalize_embeddings=True).tolist()
        col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
        logger.info(f"[memory] Миграция завершена: {col.count()} записей")
    except Exception as e:
        logger.error(f"[memory] Ошибка миграции: {e}")


# ───────────────────────────────────────────────────────────────────────────────
# Публичный API
# ───────────────────────────────────────────────────────────────────────────────

def add_fact(fact: str, category: str = "общее", source: str = "") -> bool:
    """
    Добавляет факт в векторную БД.
    Дедупликация: если cosine similarity с ближайшим фактом >= 0.92 — пропускаем.
    Возвращает True если добавлен, False если дубликат.
    """
    fact = fact.strip()
    if not fact:
        return False
    try:
        col   = _get_collection()
        embed = _get_embed_model()
        vec   = embed.encode([fact], normalize_embeddings=True).tolist()

        # проверка на дубликат (только если есть хотя бы один факт)
        if col.count() > 0:
            res = col.query(query_embeddings=vec, n_results=1,
                            include=["distances"])
            dist = res["distances"][0][0] if res["distances"] else 1.0
            # ChromaDB cosine возвращает distance = 1 - similarity
            similarity = 1.0 - dist
            if similarity >= 0.92:
                logger.debug(f"[memory] Дубликат (sim={similarity:.3f}): {fact[:60]}")
                return False

        entry_id = str(uuid.uuid4())
        meta = {
            "category":   category,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source":     source[:200] if source else "",
        }
        col.add(ids=[entry_id], embeddings=vec, documents=[fact], metadatas=[meta])
        logger.info(f"[memory] Сохранён факт: {fact[:80]}")

        # Ограничение размера: удаляем старые если > MEMORY_MAX_FACTS
        _trim_if_needed(col)
        return True
    except Exception as e:
        logger.error(f"[memory] add_fact ошибка: {e}")
        return False


def search_memory(query: str, n_results: int = MEMORY_SEARCH_FACTS) -> list[dict]:
    """
    Семантический поиск фактов по запросу.
    Возвращает список dict: {fact, category, created_at, source, score}
    Фильтрует по MEMORY_SIM_THRESHOLD.
    """
    if not query.strip():
        return []
    try:
        col   = _get_collection()
        if col.count() == 0:
            return []
        embed = _get_embed_model()
        vec   = embed.encode([query], normalize_embeddings=True).tolist()
        n     = min(n_results, col.count())
        res   = col.query(
            query_embeddings=vec,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - dist
            if score < MEMORY_SIM_THRESHOLD:
                continue
            results.append({
                "fact":       doc,
                "category":   meta.get("category", ""),
                "created_at": meta.get("created_at", ""),
                "source":     meta.get("source", ""),
                "score":      round(score, 4),
            })
        return results
    except Exception as e:
        logger.error(f"[memory] search_memory ошибка: {e}")
        return []


def get_memory_context(max_facts: int = MEMORY_CONTEXT_FACTS) -> str:
    """
    Возвращает N последних фактов в виде строки для системного prompt.
    Использует get() по офсету (без векторного поиска) — быстро.
    """
    try:
        col = _get_collection()
        n   = col.count()
        if n == 0:
            return ""
        offset = max(0, n - max_facts)
        res    = col.get(
            limit=max_facts,
            offset=offset,
            include=["documents"],
        )
        docs = res.get("documents", [])
        if not docs:
            return ""
        lines = [f"- {d}" for d in docs]
        return "Известные факты о пользователе и его жизни:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"[memory] get_memory_context ошибка: {e}")
        return ""


def get_all_facts() -> list[dict]:
    """Возвращает все факты без векторного поиска (для отладки/дампа)."""
    try:
        col = _get_collection()
        if col.count() == 0:
            return []
        res = col.get(include=["documents", "metadatas"])
        return [
            {
                "fact":       doc,
                "category":   meta.get("category", ""),
                "created_at": meta.get("created_at", ""),
                "source":     meta.get("source", ""),
            }
            for doc, meta in zip(res["documents"], res["metadatas"])
        ]
    except Exception as e:
        logger.error(f"[memory] get_all_facts ошибка: {e}")
        return []


# ───────────────────────────────────────────────────────────────────────────────
# Фоновое извлечение фактов из диалога
# ───────────────────────────────────────────────────────────────────────────────

_extract_queue: queue.Queue = queue.Queue(maxsize=32)
_extract_started = False
_extract_lock    = threading.Lock()


def _ensure_extract_thread() -> None:
    global _extract_started
    with _extract_lock:
        if _extract_started:
            return
        t = threading.Thread(target=_extract_loop, daemon=True, name="jarvis-memory-extract")
        t.start()
        _extract_started = True


def _extract_loop() -> None:
    while True:
        try:
            item = _extract_queue.get(timeout=5.0)
        except queue.Empty:
            continue
        if item is None:
            break
        user_text, jarvis_answer = item
        try:
            _extract_worker(user_text, jarvis_answer)
        except Exception as e:
            logger.error(f"[memory extract] воркер ошибка: {e}")
        finally:
            _extract_queue.task_done()


def extract_and_save_async(user_text: str, jarvis_answer: str) -> None:
    """Queue-базированное извлечение. Не блокирует."""
    _ensure_extract_thread()
    try:
        _extract_queue.put_nowait((user_text, jarvis_answer))
    except queue.Full:
        logger.warning("[memory] Очередь извлечения переполнена, пропускаю")


_EXTRACT_SYSTEM = (
    "Ты — система извлечения фактов для голосового ассистента.\n"
    "Найди ТОЛЬКО конкретные, долгосрочно полезные факты о пользователе или его жизни.\n"
    "Правила:\n"
    "1. Только факты о пользователе (имя, возраст, привычки, предпочтения, работа, планы, здоровье, интересы).\n"
    "2. НЕ записывай: погоду, курсы, новости, всё что не про него лично.\n"
    "3. Каждый факт — отдельная строка, начинается с 'ФАКТ:'.\n"
    "4. Если фактов нет — только слово 'НЕТУ'.\n"
    "5. Самодостаточные предложения, без 'он'/'она' — используй 'пользователь' или имя.\n"
    "6. Максимум 3 факта за один ход."
)


def _extract_worker(user_text: str, jarvis_answer: str) -> None:
    prompt = (
        f"Диалог:\nПользователь: {user_text[:400]}\n"
        f"Ассистент: {jarvis_answer[:400]}\n\nФакты:"
    )
    try:
        raw = chat(
            model=OLLAMA_FAST_MODEL,
            messages=[
                {"role": "system",  "content": _EXTRACT_SYSTEM},
                {"role": "user",    "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 150, "num_ctx": 2048},
        )
    except Exception as e:
        logger.error(f"[memory extract] ошибка LLM: {e}")
        return

    if not raw or "НЕТУ" in raw.upper():
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line.upper().startswith("ФАКТ:"):
            continue
        fact = line[5:].strip()
        if len(fact) < 8:
            continue
        add_fact(fact=fact, category="авто", source=user_text[:100])


# ───────────────────────────────────────────────────────────────────────────────
# Внутренние утилиты
# ───────────────────────────────────────────────────────────────────────────────

def _trim_if_needed(col: Any) -> None:
    """Удаляет самые старые записи если превышен MEMORY_MAX_FACTS."""
    try:
        count = col.count()
        if count <= MEMORY_MAX_FACTS:
            return
        excess = count - MEMORY_MAX_FACTS
        # получаем самые старые (offset=0)
        res    = col.get(limit=excess, offset=0, include=[])
        old_ids = res.get("ids", [])
        if old_ids:
            col.delete(ids=old_ids)
            logger.info(f"[memory] Удалено {len(old_ids)} старых фактов (превышен лимит {MEMORY_MAX_FACTS})")
    except Exception as e:
        logger.warning(f"[memory] _trim_if_needed ошибка: {e}")
