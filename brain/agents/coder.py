"""
brain/agents/coder.py — CoderAgent (Level 4).

Один файл за вызов. Принимает:
  spec   — дистил intake-спеки проекта (dict)
  plan   — архитектурный план (dict)
  target — описание файла {"path", "purpose", "depends_on"}
  feedback — опциональные замечания от ReviewerAgent (str), пусто на первой итерации

Возвращает чистый код файла (str). Никогда не падает молча — на ошибке
LLM возвращает короткий комментарий-заглушку, чтобы оркестратор увидел проблему.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from brain.client import chat, MODEL_FAST, MODEL_HEAVY
from brain.prompts import PROJECT_CODER_SYSTEM

logger = logging.getLogger(__name__)

CODER_TEMPERATURE = 0.2
CODER_NUM_CTX     = 8192


def _strip_code_fence(raw: str) -> str:
    """LLMs often wrap code in ```python ... ``` despite the instruction."""
    s = raw.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.rstrip() + "\n"


MAX_CTX_FILE_BYTES = 4000  # лимит на один соседний файл в контексте
MAX_CTX_TOTAL_BYTES = 12000  # общий лимит на cross-file context


def _format_existing_files(existing: dict[str, str] | None, current_path: str) -> str:
    if not existing:
        return ""
    chunks = []
    total = 0
    for path, content in existing.items():
        if path == current_path:
            continue
        body = (content or "")[:MAX_CTX_FILE_BYTES]
        if not body.strip():
            continue
        chunk = f"### {path}\n```\n{body}\n```\n"
        if total + len(chunk) > MAX_CTX_TOTAL_BYTES:
            chunks.append("### … (остальные файлы урезаны по бюджету)\n")
            break
        chunks.append(chunk)
        total += len(chunk)
    if not chunks:
        return ""
    return "\nУже написанные файлы проекта (используй их API корректно):\n" + "".join(chunks)


def _build_user_message(spec: dict, plan: dict, target: dict, feedback: str,
                        existing: dict[str, str] | None = None) -> str:
    files_outline = "\n".join(
        f"  - {f.get('path')}: {f.get('purpose','')}"
        for f in (plan.get("files") or [])
    ) or "  (нет других файлов)"

    msg = (
        f"Проект: {spec.get('title','?')}\n"
        f"Тип: {spec.get('kind','script')}, язык: {spec.get('language','python')}\n"
        f"Суть: {spec.get('summary','')}\n\n"
        f"Требования:\n" + "\n".join(f"  - {r}" for r in (spec.get('requirements') or [])) + "\n\n"
        f"Acceptance criteria:\n" + "\n".join(f"  - {a}" for a in (spec.get('acceptance_criteria') or [])) + "\n\n"
        f"Структура проекта:\n{files_outline}\n"
    )
    msg += _format_existing_files(existing, target.get("path", ""))
    msg += (
        f"\nСейчас тебе нужно написать ОДИН файл:\n"
        f"  путь:    {target.get('path')}\n"
        f"  цель:    {target.get('purpose','')}\n"
        f"  зависит: {', '.join(target.get('depends_on') or []) or 'stdlib'}\n"
    )
    if feedback:
        msg += (
            f"\nReviewer попросил исправить следующие проблемы:\n{feedback}\n"
            f"Перепиши файл целиком с учётом замечаний.\n"
        )
    return msg


def write_file(
    spec: dict,
    plan: dict,
    target: dict,
    feedback: str = "",
    *,
    existing: dict[str, str] | None = None,
    model: str = MODEL_FAST,
) -> str:
    """Generate full file content. Falls back to stub on error."""
    if not isinstance(target, dict) or "path" not in target:
        raise ValueError("target must have 'path'")
    msgs = [
        {"role": "system", "content": PROJECT_CODER_SYSTEM},
        {"role": "user",   "content": _build_user_message(spec, plan, target, feedback, existing)},
    ]
    try:
        raw = chat(model, msgs, options={"temperature": CODER_TEMPERATURE, "num_ctx": CODER_NUM_CTX})
    except Exception as e:
        logger.error(f"[coder] LLM error for {target.get('path')}: {e}")
        return (
            f"# AUTO-GENERATED STUB — coder LLM call failed: {e}\n"
            f"# path: {target.get('path')}\n"
            f"raise NotImplementedError('coder failed: see manifest phase log')\n"
        )
    code = _strip_code_fence(raw)
    if not code.strip():
        logger.warning(f"[coder] empty output for {target.get('path')} — stubbing")
        return f"# AUTO-GENERATED STUB — empty LLM output\n"
    return code


def patch_file(
    spec: dict,
    plan: dict,
    target: dict,
    current_code: str,
    feedback: str,
    *,
    existing: dict[str, str] | None = None,
    model: str = MODEL_FAST,
) -> str:
    """
    Re-generate file content given current code + reviewer feedback.
    For now, regeneration mode (LLM rewrites whole file). Diff-based patching
    is a future iteration.
    """
    user_msg = _build_user_message(spec, plan, target, feedback, existing)
    user_msg += (
        "\n\nТекущая версия файла (требует исправления):\n"
        f"```\n{current_code}\n```\n"
    )
    msgs = [
        {"role": "system", "content": PROJECT_CODER_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    try:
        raw = chat(model, msgs, options={"temperature": CODER_TEMPERATURE, "num_ctx": CODER_NUM_CTX})
    except Exception as e:
        logger.error(f"[coder.patch] LLM error: {e}")
        return current_code   # лучше не ломать рабочий файл
    return _strip_code_fence(raw) or current_code
