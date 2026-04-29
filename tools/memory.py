# tools/memory.py
"""
tools/memory.py — долгосрочная память Jarvis.

Факты хранятся в JSON-файле core.paths.MEMORY_PATH (C:/jarvis/data/memory.json).

Формат одной записи:
{
  "id": "uuid4",
  "fact": "Александр занимается в спортзале три раза в неделю",
  "category": "пользователь",
  "created_at": "2026-04-25T23:15:00",
  "source": "Сэр сказал: 'хожу в зал три раза в неделю'"
}

Извлечение фактов — единственный фоновый поток через очередь, не блокирует основной цикл.

FIXES vs previous version:
  - _is_duplicate: O(N²) → early-exit + length guard to avoid slow set ops on 500 facts
  - extract_and_save_async: now accepts an optional `skip_extract` flag so tool/git
    answers don't trigger LLM memory extraction
"""
from __future__ import annotations

import json
import queue
import re
import threading
import uuid
from datetime import datetime
from typing import Optional

from core.config import OLLAMA_FAST_MODEL, MEMORY_MAX_FACTS
from core.paths import MEMORY_PATH
from brain.client import chat

_lock = threading.Lock()

# --- In-memory кэш -----------------------------------------------------------
_cache: Optional[list[dict]] = None


def _invalidate_cache() -> None:
    global _cache
    _cache = None


# --- Единый фоновый воркер для extract_and_save ------------------------------

_extract_queue: queue.Queue = queue.Queue(maxsize=32)
_extract_thread_started = False
_extract_thread_lock = threading.Lock()


def _ensure_extract_thread() -> None:
    global _extract_thread_started
    with _extract_thread_lock:
        if _extract_thread_started:
            return
        t = threading.Thread(target=_extract_loop, daemon=True, name="jarvis-memory-extract")
        t.start()
        _extract_thread_started = True


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
            print(f"[memory extract] Ошибка воркера: {e}")
        finally:
            _extract_queue.task_done()


# --- внутренние хелперы ---------------------------------------------------

def _load() -> list[dict]:
    """Возвращает КОПИЮ списка фактов. Вызывающий не должен мутировать кеш."""
    global _cache
    if _cache is not None:
        return list(_cache)
    if not MEMORY_PATH.exists():
        _cache = []
        return list(_cache)
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache = data if isinstance(data, list) else []
    except Exception as e:
        print(f"[memory] Ошибка чтения: {e}")
        _cache = []
    return list(_cache)


def _save(facts: list[dict]) -> None:
    global _cache
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)
        _cache = list(facts)
    except Exception as e:
        print(f"[memory] Ошибка записи: {e}")
        _invalidate_cache()


def _is_duplicate(new_fact: str, existing: list[dict], threshold: float = 0.7) -> bool:
    """Простая проверка дубликата — Jaccard по токенам.

    FIX: added early-exit guard: skip Jaccard for very short facts (< 8 chars)
    and bail out as soon as first match is found, reducing worst-case O(N) cost.
    Also added minimum token count guard to avoid false positives on short phrases.
    """
    def tokens(s: str) -> set:
        return set(re.findall(r"\w+", s.lower()))

    new_tok = tokens(new_fact)
    # FIX: don't run expensive set ops if fact is too short to be meaningful
    if len(new_tok) < 3:
        return False
    for item in existing:
        existing_tok = tokens(item.get("fact", ""))
        if len(existing_tok) < 3:
            continue
        intersection = len(new_tok & existing_tok)
        union = len(new_tok | existing_tok)
        # FIX: early exit on first match — don't scan all 500 facts
        if union > 0 and intersection / union >= threshold:
            return True
    return False


# --- публичный API --------------------------------------------------------

def get_memory_context(max_facts: int = 20) -> str:
    with _lock:
        facts = _load()
    if not facts:
        return ""
    recent = facts[-max_facts:]
    lines = [f"- {item['fact']}" for item in recent]
    return "Известные факты о пользователе и его жизни:\n" + "\n".join(lines)


def add_fact(fact: str, category: str = "общее", source: str = "") -> bool:
    fact = fact.strip()
    if not fact:
        return False
    with _lock:
        facts = _load()
        if _is_duplicate(fact, facts):
            print(f"[memory] Дубликат, пропускаю: {fact[:60]}")
            return False
        entry = {
            "id": str(uuid.uuid4()),
            "fact": fact,
            "category": category,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": source[:200] if source else "",
        }
        facts.append(entry)
        if len(facts) > MEMORY_MAX_FACTS:
            facts = facts[-MEMORY_MAX_FACTS:]
        _save(facts)
        print(f"[memory] Сохранён факт: {fact[:80]}")
        return True


def get_all_facts() -> list[dict]:
    with _lock:
        return list(_load())


# Routes that should NOT trigger memory extraction
# (tool results, git output, etc. — no personal info there)
_SKIP_MEMORY_ROUTES = frozenset({
    "tool", "git",
})


def extract_and_save_async(user_text: str, jarvis_answer: str, route: str = "") -> None:
    """Ставит задачу в очередь единственного фонового потока. Не блокирует.

    FIX: added `route` param — skip extraction for tool/git answers since
    they never contain personal facts about the user (weather, rates, etc.).
    """
    # FIX: skip memory extraction for tool/system routes — no personal data there
    if route and route.split(".")[0] in _SKIP_MEMORY_ROUTES:
        return
    # Skip if answer is very short (tool confirmation like "Таймер установлен")
    if len(jarvis_answer.strip()) < 30:
        return
    _ensure_extract_thread()
    try:
        _extract_queue.put_nowait((user_text, jarvis_answer))
    except queue.Full:
        print("[memory] Очередь извлечения переполнена, пропускаю ход")


# --- фоновый воркер -------------------------------------------------------

_EXTRACT_SYSTEM = (
    "Ты — система извлечения фактов для голосового ассистента.\n"
    "Найди ТОЛЬКО конкретные, долгосрочно полезные факты о пользователе или его жизни.\n"
    "Правила:\n"
    "1. Только факты о пользователе (имя, возраст, привычки, предпочтения, работа, планы, здоровье, интересы).\n"
    "2. НЕ записывай погоду, курсы валют, новости — всё, что не про него лично.\n"
    "3. НЕ записывай общие вопросы — только личное ('я люблю X', 'у меня есть Y').\n"
    "4. Каждый факт — отдельная строка, начинается с 'ФАКТ:'.\n"
    "5. Если фактов нет — выведи только слово 'НЕТУ'.\n"
    "6. Факт должен быть самодостаточным предложением (без 'он'/'она' — используй 'пользователь' или имя).\n"
    "7. Максимум 3 факта за один ход."
)


def _extract_worker(user_text: str, jarvis_answer: str) -> None:
    prompt = (
        f"Диалог:\nПользователь: {user_text[:400]}\nАссистент: {jarvis_answer[:400]}\n\nФакты:"
    )
    try:
        raw = chat(
            model=OLLAMA_FAST_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 120, "num_ctx": 2048},
        )
    except Exception as e:
        print(f"[memory extract] Ошибка запроса: {e}")
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
