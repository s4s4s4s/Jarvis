"""
brain/agents/project_clarify.py — обязательный этап уточнения требований для ProjectAgent.

Принцип:
- Jarvis НЕ угадывает функционал проекта, если в запросе есть существенные неопределённости.
- Сначала задаёт ВСЕ ключевые уточняющие вопросы одним сообщением.
- Только после ответов пользователя формирует spec/intake.
- Лишь затем, если нужны токены/ключи, запрашивает credentials.

Workflow:
  1. project.run(query) вызывает maybe_start_clarify(query).
  2. Если запрос неполный → возвращается список вопросов и сохраняется pending state.
  3. Следующее сообщение пользователя → ask.py видит pending clarify и вызывает provide_clarify_answers().
  4. Ответы склеиваются с исходным запросом и отправляются обратно в project.run(...).
  5. Уже после этого возможен credentials step.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from brain.client import chat, MODEL_FAST
from brain.prompts import PROJECT_CLARIFY_SYSTEM

logger = logging.getLogger(__name__)


def _data_dir() -> Path:
    from core.paths import DATA_DIR
    return Path(DATA_DIR)


def pending_clarify_path() -> Path:
    return _data_dir() / "projects" / "_pending_clarify.json"


def save_pending_clarify(query: str, questions_text: str) -> None:
    path = pending_clarify_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"query": query, "questions_text": questions_text}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[project_clarify] save failed: {e}")


def load_pending_clarify() -> dict | None:
    path = pending_clarify_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[project_clarify] load failed: {e}")
        return None


def clear_pending_clarify() -> None:
    try:
        pending_clarify_path().unlink(missing_ok=True)
    except Exception:
        pass


def has_pending_clarify() -> bool:
    return pending_clarify_path().exists()


def maybe_start_clarify(query: str) -> str | None:
    """Проверяет, нужны ли уточняющие вопросы.

    Возвращает:
      - str с вопросами, если проект нельзя качественно реализовать без уточнений
      - None, если запрос уже достаточно конкретный
    """
    prompt = (
        "Запрос пользователя на разработку проекта:\n\n"
        f"{query.strip()}\n\n"
        "Либо верни EXACT_OK если запрос уже достаточно определён, "
        "либо верни список ВСЕХ критичных уточняющих вопросов одним сообщением."
    )
    try:
        raw = chat(MODEL_FAST, [
            {"role": "system", "content": PROJECT_CLARIFY_SYSTEM},
            {"role": "user", "content": prompt},
        ], options={"temperature": 0.1, "num_ctx": 4096})
    except Exception as e:
        logger.warning(f"[project_clarify] LLM failed: {e}")
        return None

    text = (raw or "").strip()
    if not text or text == "EXACT_OK":
        return None

    save_pending_clarify(query, text)
    return text


def provide_clarify_answers(answer_text: str) -> str:
    """Принимает ответы пользователя на уточняющие вопросы и перезапускает ProjectAgent."""
    pending = load_pending_clarify()
    if not pending:
        return "Нет проекта, ожидающего уточнений."

    original_query = str(pending.get("query") or "").strip()
    clear_pending_clarify()

    merged_query = (
        f"{original_query}\n\n"
        "Уточнения пользователя по проекту:\n"
        f"{(answer_text or '').strip()}"
    )

    try:
        from brain.agents.project import run
        return run(merged_query, history=[])
    except Exception as e:
        return f"Уточнения получены, но перезапустить проект не удалось: {e}"


def looks_like_clarify_answer(text: str) -> bool:
    """Мягкая эвристика: любой непустой осмысленный ответ принимаем как ответы на вопросы.

    Здесь это допустимо, потому что решение не о маршрутизации всего ассистента,
    а только о продолжении уже явно pending-сценария уточнения проекта.
    """
    s = (text or "").strip()
    if not s:
        return False
    if re.search(r"[A-Za-zА-Яа-я0-9]", s) is None:
        return False
    return True
