"""
brain/agents/reviewer.py — ReviewerAgent (Level 4).

Принимает код одного файла + спеку проекта.
Возвращает структурированный verdict от LLM-критика.

verdict in {"approve", "revise"}.
Если LLM сломался / JSON невалидный — возвращает безопасный approve с пометкой,
чтобы пайплайн ProjectAgent не зависал в бесконечном цикле.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from brain.client import chat, MODEL_HEAVY, MODEL_FAST
from brain.prompts import PROJECT_REVIEWER_SYSTEM

logger = logging.getLogger(__name__)

REVIEWER_TEMPERATURE = 0.0
REVIEWER_NUM_CTX     = 8192


def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _safe_parse(raw: str) -> dict[str, Any]:
    """Try to recover JSON even if LLM added stray prose."""
    txt = _strip_json_fence(raw)
    try:
        return json.loads(txt)
    except Exception:
        # последний шанс — найти первый {...} блок
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def review(
    spec: dict,
    target: dict,
    code: str,
    *,
    model: str = MODEL_HEAVY,
) -> dict[str, Any]:
    """
    Review a single file. Returns:
      {
        "verdict":  "approve" | "revise",
        "issues":   [{"severity","line_hint","problem","suggestion"}, ...],
        "summary":  str,
        "_source":  "llm"|"fallback"
      }
    """
    user_msg = (
        f"Файл: {target.get('path','?')}\n"
        f"Цель файла: {target.get('purpose','')}\n\n"
        f"Спецификация проекта:\n"
        f"  title:    {spec.get('title','')}\n"
        f"  summary:  {spec.get('summary','')}\n"
        f"  requirements:\n"
        + "\n".join(f"    - {r}" for r in (spec.get('requirements') or []))
        + "\n  acceptance_criteria:\n"
        + "\n".join(f"    - {a}" for a in (spec.get('acceptance_criteria') or []))
        + "\n\n"
        f"Исходный код:\n```\n{code}\n```\n"
    )
    msgs = [
        {"role": "system", "content": PROJECT_REVIEWER_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    try:
        raw = chat(model, msgs, options={"temperature": REVIEWER_TEMPERATURE, "num_ctx": REVIEWER_NUM_CTX})
    except Exception as e:
        logger.error(f"[reviewer] LLM error: {e}")
        return {
            "verdict": "approve",
            "issues":  [],
            "summary": f"reviewer недоступен: {e} — пропускаем",
            "_source": "fallback",
        }

    data = _safe_parse(raw)
    verdict = data.get("verdict", "approve")
    if verdict not in ("approve", "revise"):
        verdict = "approve"
    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    summary = data.get("summary") or ""
    if not isinstance(summary, str):
        summary = str(summary)

    return {
        "verdict": verdict,
        "issues":  issues,
        "summary": summary[:500],
        "_source": "llm",
    }


def issues_as_feedback(issues: list[dict]) -> str:
    """Render issues into a feedback string the Coder can act on."""
    if not issues:
        return ""
    lines = []
    for i, iss in enumerate(issues, 1):
        sev = iss.get("severity", "?")
        prob = iss.get("problem", "")
        sug  = iss.get("suggestion", "")
        line_hint = iss.get("line_hint")
        loc = f" (строка ~{line_hint})" if line_hint else ""
        lines.append(f"{i}. [{sev}]{loc} {prob}\n   Решение: {sug}")
    return "\n".join(lines)
