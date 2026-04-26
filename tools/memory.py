# C:\jarvis\tools\memory.py
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

Извлечение фактов — фоновый поток, не блокирует основной цикл.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import requests

from core.config import OLLAMA_URL, MEMORY_MAX_FACTS
from core.paths import MEMORY_PATH

_lock = threading.Lock()

# Для извлечения фактов используем fast-модель.
OLLAMA_EXTRACT_MODEL = "llama3.1:8b-instruct-q5_K_M"


# --- внутренние хелперы ---------------------------------------------------

def _load() -> list[dict]:
    p = Path(MEMORY_PATH)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[memory] Ошибка чтения: {e}")
        return []


def _save(facts: list[dict]) -> None:
    p = Path(MEMORY_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[memory] Ошибка записи: {e}")


def _is_duplicate(new_fact: str, existing: list[dict], threshold: float = 0.7) -> bool:
    """Простая проверка дубликата — нормализованное пересечение слов (Jaccard)."""
    def tokens(s: str) -> set:
        return set(re.findall(r"\w+", s.lower()))

    new_tok = tokens(new_fact)
    if not new_tok:
        return False

    for item in existing:
        existing_tok = tokens(item.get("fact", ""))
        if not existing_tok:
            continue
        intersection = len(new_tok & existing_tok)
        union = len(new_tok | existing_tok)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


# --- публичный API --------------------------------------------------------

def get_memory_context(max_facts: int = 20) -> str:
    """
    Возвращает строку для вставки в системный промпт LLM.
    Берёт последние max_facts записей.
    """
    with _lock:
        facts = _load()

    if not facts:
        return ""

    recent = facts[-max_facts:]
    lines = [f"- {item['fact']}" for item in recent]
    block = "\n".join(lines)
    return f"Известные факты о пользователе и его жизни:\n{block}"


def add_fact(fact: str, category: str = "общее", source: str = "") -> bool:
    """
    Добавляет факт вручную (например, из команды «запомни, что…»).
    Возвращает True если добавлен, False если дубликат или пустая строка.
    """
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
    """Возвращает все сохранённые факты (для отладки / будущего UI)."""
    with _lock:
        return list(_load())


def extract_and_save_async(user_text: str, jarvis_answer: str) -> None:
    """
    Запускает фоновый поток: просит LLM извлечь факты из диалогового хода
    и сохраняет их. Не блокирует основной поток.
    """
    t = threading.Thread(
        target=_extract_worker,
        args=(user_text, jarvis_answer),
        daemon=True,
    )
    t.start()


# --- фоновый воркер -------------------------------------------------------

_EXTRACT_PROMPT = """\
Ты — система извлечения фактов для голосового ассистента.

Тебе дан один ход диалога между пользователем (Сэром) и ассистентом.
Твоя задача: найти ТОЛЬКО конкретные, долгосрочно полезные факты о пользователе или его жизни.

Правила:
1. Выводи ТОЛЬКО факты о пользователе (имя, возраст, привычки, предпочтения, работа, планы, здоровье, интересы, важные события).
2. НЕ записывай погоду, курсы валют, новости — всё, что не про него лично.
3. НЕ записывай общие вопросы («что такое X?») — только личное («я люблю X», «у меня есть Y»).
4. Каждый факт — отдельная строка, начинается с «ФАКТ:».
5. Если фактов нет — выведи только слово «НЕТУ».
6. Факт должен быть самодостаточным предложением (без «он», «она» — используй «пользователь» или имя).
7. Максимум 3 факта за один ход.

Диалог:
Пользователь: {user}
Ассистент: {jarvis}

Факты:"""


def _extract_worker(user_text: str, jarvis_answer: str) -> None:
    prompt = _EXTRACT_PROMPT.format(
        user=user_text[:400],
        jarvis=jarvis_answer[:400],
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_EXTRACT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 120,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[memory extract] Ошибка запроса: {e}")
        return

    if "НЕТУ" in raw.upper() or not raw:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line.upper().startswith("ФАКТ:"):
            continue
        fact = line[5:].strip()
        if len(fact) < 8:
            continue
        add_fact(
            fact=fact,
            category="авто",
            source=f"{user_text[:100]}",
        )
# === end of file: tools/memory.py ===