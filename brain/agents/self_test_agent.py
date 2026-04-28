"""brain/agents/self_test_agent.py

Self-Test Agent — Джарвис тестирует себя сам.

Поток:
  1. LLM генерирует N реальных запросов, покрывающих все route-типы
  2. Каждый запрос прогоняется через полный pipeline ask_llm()
  3. LLM-аудитор оценивает пару (query, response): pass/fail + issues + suggestions
  4. Результаты сохраняются в logs/self_test_<ts>.json

Формат каждой записи (готов для fine-tuning):
  {
    "timestamp": "...",
    "query": "...",
    "expected_route": "plan",
    "actual_route": "plan",
    "response": "...",
    "elapsed_ms": 4200,
    "verdict": "pass" | "fail",
    "issues": ["..."],
    "suggestions": ["..."]
  }
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.client import chat, MODEL_HEAVY, MODEL_ROUTER

logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")

# Все route-типы, которые должны быть покрыты хотя бы одним тестом
ALL_ROUTES = ["chat", "code", "plan", "web", "tool", "memory", "deep"]

_GENERATE_SYSTEM = """\
You are a test-case designer for an AI assistant called Jarvis.
Jarvis routes user messages to one of these agents:
  chat    — casual conversation, general questions
  code    — write, run or fix a Python script
  plan    — multi-step task that needs planning + execution
  web     — search the internet for current information
  tool    — live data: weather, crypto price, currency rate, timer, time
  memory  — recall something from previous conversations
  deep    — single heavy analytical / reasoning question

Your job: generate realistic, diverse test queries that a real user might send.
Each query must clearly target ONE of the routes above.

Rules:
- Queries must be in Russian (that is the user's language).
- Each query must be a complete, natural sentence — not a category label.
- Cover ALL routes listed above at least once.
- If asked for more tests than 7, add extra queries varying difficulty and phrasing.
- Return ONLY a valid JSON array, no markdown, no extra text.

Each element:
  {"query": "<natural Russian sentence>", "expected_route": "<route>"}
"""

_AUDIT_SYSTEM = """\
You are a strict QA auditor evaluating an AI assistant called Jarvis.
You receive:
  - The original user query
  - The route Jarvis chose (intended agent)
  - Jarvis's full response

Your job: decide if Jarvis FULLY and CORRECTLY handled the query.

Criteria:
  1. Did Jarvis understand what was asked?
  2. Is the response complete and accurate for that type of request?
  3. Are there errors, hallucinations, missing parts, or wrong format?
  4. For code/plan routes: is the output actually useful / runnable?
  5. For tool routes: did it return real structured data?
  6. For web routes: did it actually answer with retrieved information?
  7. For memory routes: did it recall relevant context?

Respond with ONLY a valid JSON object, no markdown:
{
  "verdict": "pass" | "fail",
  "score": <0.0 to 1.0>,
  "issues": ["<issue1>", "<issue2>"],
  "suggestions": ["<suggestion1>"]
}

If verdict is "pass" and no issues, issues and suggestions may be empty arrays.
"""


def _extract_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner, in_block = [], False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True; continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        raw = "\n".join(inner).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found: {raw[:200]}")
    return json.loads(raw[start:end + 1])


def _extract_json_object(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner, in_block = [], False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True; continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        raw = "\n".join(inner).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found: {raw[:200]}")
    return json.loads(raw[start:end + 1])


def _parse_n_from_query(query: str) -> int | None:
    """Extract explicit test count from query, e.g. 'прогони 15 тестов' → 15."""
    m = re.search(r"(\d+)\s*тест", query)
    return int(m.group(1)) if m else None


def _generate_test_cases(n: int) -> list[dict]:
    """Ask LLM to generate n test cases covering all routes."""
    prompt = (
        f"Generate exactly {n} test queries. "
        f"Cover ALL 7 routes at least once. "
        f"If {n} > 7, add more varied queries for existing routes."
    )
    messages = [
        {"role": "system", "content": _GENERATE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = chat(model=MODEL_HEAVY, messages=messages, options={"temperature": 0.7})
    cases = _extract_json_array(raw)
    # валидация структуры
    valid = []
    for c in cases:
        if isinstance(c, dict) and "query" in c and "expected_route" in c:
            valid.append({"query": str(c["query"]), "expected_route": str(c["expected_route"])})
    logger.info("[SelfTest] Generated %d test cases", len(valid))
    return valid


def _run_single_test(case: dict) -> dict:
    """Run one test case through the full Jarvis pipeline and audit the result."""
    query = case["query"]
    expected_route = case["expected_route"]

    logger.info("[SelfTest] Running: %s (expected=%s)", query[:80], expected_route)

    # --- Прогон через полный pipeline ---
    t0 = time.monotonic()
    actual_route = "unknown"
    response = ""
    pipeline_error: str | None = None

    try:
        from brain.ask import ask_llm, _route
        # Сначала узнаём реальный route
        route_data = _route(query, history=[])
        actual_route = route_data.get("route", "unknown")
        # Полный ответ
        result = ask_llm(query)
        response = result.get_answer(timeout=180.0)
    except Exception as e:
        pipeline_error = str(e)
        response = f"[PIPELINE ERROR] {e}"
        logger.error("[SelfTest] Pipeline error for query '%s': %s", query[:60], e)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # --- Аудит LLM ---
    audit_result: dict[str, Any] = {
        "verdict": "fail",
        "score": 0.0,
        "issues": [f"Pipeline error: {pipeline_error}"] if pipeline_error else [],
        "suggestions": [],
    }

    if not pipeline_error:
        try:
            audit_messages = [
                {"role": "system", "content": _AUDIT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n"
                        f"Route used: {actual_route}\n"
                        f"Expected route: {expected_route}\n\n"
                        f"Jarvis response:\n{response[:4000]}"
                    ),
                },
            ]
            raw_audit = chat(
                model=MODEL_HEAVY,
                messages=audit_messages,
                options={"temperature": 0.1},
            )
            parsed = _extract_json_object(raw_audit)
            audit_result = {
                "verdict":     parsed.get("verdict", "fail"),
                "score":       float(parsed.get("score", 0.0)),
                "issues":      parsed.get("issues", []),
                "suggestions": parsed.get("suggestions", []),
            }
        except Exception as e:
            logger.error("[SelfTest] Auditor failed for query '%s': %s", query[:60], e)
            audit_result["issues"].append(f"Auditor error: {e}")

    record = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "query":          query,
        "expected_route": expected_route,
        "actual_route":   actual_route,
        "route_match":    actual_route == expected_route,
        "response":       response,
        "elapsed_ms":     elapsed_ms,
        "verdict":        audit_result["verdict"],
        "score":          audit_result["score"],
        "issues":         audit_result["issues"],
        "suggestions":    audit_result["suggestions"],
    }

    status = "✅" if audit_result["verdict"] == "pass" else "❌"
    route_match_str = "✔" if record["route_match"] else f"✘ (got {actual_route})"
    logger.info(
        "[SelfTest] %s  route=%s  score=%.2f  '%s'",
        status, route_match_str, audit_result["score"], query[:60],
    )
    return record


def _save_log(records: list[dict], run_id: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"self_test_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("[SelfTest] Log saved: %s", log_path)
    return log_path


def _build_summary(records: list[dict]) -> str:
    total   = len(records)
    passed  = sum(1 for r in records if r["verdict"] == "pass")
    failed  = total - passed
    avg_score = sum(r["score"] for r in records) / total if total else 0.0
    route_mismatches = [r for r in records if not r["route_match"]]
    fail_records = [r for r in records if r["verdict"] == "fail"]

    lines = [
        f"✅ Прошло: {passed}/{total}   "
        f"❌ Упало: {failed}/{total}   "
        f"⭐ Ср. скор: {avg_score:.2f}",
    ]

    if route_mismatches:
        lines.append("\n⚠️  Неверный route:")
        for r in route_mismatches:
            lines.append(
                f"  • [{r['expected_route']} → {r['actual_route']}] «{r['query'][:70]}»"
            )

    if fail_records:
        lines.append("\n❌  Проваленные тесты:")
        for r in fail_records[:5]:  # показываем макс 5 чтобы не заспамить ответ
            lines.append(f"  «{r['query'][:70]}»")
            for issue in r["issues"][:2]:
                lines.append(f"    — {issue}")
            for sug in r["suggestions"][:1]:
                lines.append(f"    → {sug}")

    return "\n".join(lines)


def run(query: str, history: list[dict] | None = None) -> str:  # noqa: ARG001
    """Entry point for route='test'."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Определяем количество тестов
    explicit_n = _parse_n_from_query(query)
    n = explicit_n if explicit_n and explicit_n > 0 else len(ALL_ROUTES)  # default: 7
    n = min(n, 50)  # жёсткий лимит сверху

    logger.info("[SelfTest] Starting run_id=%s  n=%d tests", run_id, n)

    # Генерация тестовых запросов
    try:
        cases = _generate_test_cases(n)
    except Exception as e:
        logger.error("[SelfTest] Failed to generate test cases: %s", e)
        return f"Не удалось сгенерировать тестовые запросы: {e}"

    if not cases:
        return "Ошибка: LLM не вернул ни одного тестового запроса."

    # Прогон тестов
    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        logger.info("[SelfTest] Test %d/%d", i, len(cases))
        record = _run_single_test(case)
        records.append(record)

    # Сохранение лога
    log_path = _save_log(records, run_id)

    # Отчёт пользователю
    summary = _build_summary(records)
    return (
        f"🧠 Самотестирование завершено [{run_id}]\n\n"
        + summary
        + f"\n\n📄 Лог: {log_path.resolve()}"
    )
