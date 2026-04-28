"""brain/agents/self_test_agent.py

Self-Test Agent.

Each test calls report_progress() so the UI gets live updates
without waiting for the full run to finish.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.client import chat, MODEL_HEAVY
from brain.ask import report_progress

logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")

ALL_ROUTES = ["chat", "code", "plan", "web", "tool", "memory", "deep"]

_GENERATE_SYSTEM = (
    "You are a test-case designer for an AI assistant called Jarvis.\n"
    "Jarvis routes user messages to one of these agents:\n"
    "  chat    - casual conversation, general questions\n"
    "  code    - write, run or fix a Python script\n"
    "  plan    - multi-step task that needs planning + execution\n"
    "  web     - search the internet for current information\n"
    "  tool    - live data: weather, crypto price, currency rate, timer, time\n"
    "  memory  - recall something from previous conversations\n"
    "  deep    - single heavy analytical / reasoning question\n"
    "\n"
    "Your job: generate realistic, diverse test queries that a real user might send.\n"
    "Each query must clearly target ONE of the routes above.\n"
    "\n"
    "Rules:\n"
    "- Queries must be in Russian (that is the user language).\n"
    "- Each query must be a complete, natural sentence, not a category label.\n"
    "- Cover ALL routes listed above at least once.\n"
    "- Return ONLY a valid JSON array, no markdown, no extra text.\n"
    "\n"
    "Each element: {\"query\": \"<natural Russian sentence>\", \"expected_route\": \"<route>\"}"
)

_AUDIT_SYSTEM = (
    "You are a strict QA auditor evaluating an AI assistant called Jarvis.\n"
    "You receive the original user query, the route Jarvis chose, and Jarvis full response.\n"
    "\n"
    "Decide if Jarvis FULLY and CORRECTLY handled the query.\n"
    "Respond with ONLY a valid JSON object, no markdown:\n"
    "{\"verdict\": \"pass\" | \"fail\", \"score\": <0.0-1.0>, "
    "\"issues\": [...], \"suggestions\": [...]}"
)


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
    m = re.search(r"(\d+)\s*тест", query)
    return int(m.group(1)) if m else None


def _generate_test_cases(n: int) -> list[dict]:
    prompt = (
        f"Generate exactly {n} test queries. "
        f"Cover ALL 7 routes at least once."
    )
    messages = [
        {"role": "system", "content": _GENERATE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = chat(model=MODEL_HEAVY, messages=messages, options={"temperature": 0.7})
    cases = _extract_json_array(raw)
    valid = [
        {"query": str(c["query"]), "expected_route": str(c["expected_route"])}
        for c in cases
        if isinstance(c, dict) and "query" in c and "expected_route" in c
    ]
    logger.info("[SelfTest] Generated %d test cases", len(valid))
    return valid


def _run_single_test(case: dict, test_num: int, total: int) -> dict:
    query = case["query"]
    expected_route = case["expected_route"]

    # Live progress: notify UI that this test is starting
    report_progress(f"⏳ Тест {test_num}/{total}: {query[:80]}")
    logger.info("[SelfTest] Test %d/%d - %s (expected=%s)", test_num, total, query[:70], expected_route)

    t0 = time.monotonic()
    actual_route = "unknown"
    response = ""
    pipeline_error: str | None = None

    try:
        from brain.ask import _route, _dispatch
        route_data = _route(query, history=[])
        actual_route = route_data.get("route", "unknown")
        response = _dispatch(route_data, query, history=[])
    except Exception as e:
        pipeline_error = str(e)
        response = f"[PIPELINE ERROR] {e}"
        logger.error("[SelfTest] Pipeline error for '%s': %s", query[:60], e)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

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
                {"role": "user", "content": (
                    f"Query: {query}\n"
                    f"Route used: {actual_route}\n"
                    f"Expected route: {expected_route}\n\n"
                    f"Jarvis response:\n{response[:4000]}"
                )},
            ]
            raw_audit = chat(model=MODEL_HEAVY, messages=audit_messages, options={"temperature": 0.1})
            parsed = _extract_json_object(raw_audit)
            audit_result = {
                "verdict":     parsed.get("verdict", "fail"),
                "score":       float(parsed.get("score", 0.0)),
                "issues":      parsed.get("issues", []),
                "suggestions": parsed.get("suggestions", []),
            }
        except Exception as e:
            logger.error("[SelfTest] Auditor failed for '%s': %s", query[:60], e)
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

    # Live progress: notify UI of result
    status = "✅" if audit_result["verdict"] == "pass" else "❌"
    route_ok = "✔" if record["route_match"] else f"✘ got={actual_route}"
    report_progress(
        f"{status} {test_num}/{total}  route={route_ok}  "
        f"score={audit_result['score']:.2f}  {elapsed_ms // 1000}s  «{query[:55]}»"
    )
    logger.info(
        "[SelfTest] %s %d/%d  route=%s  score=%.2f  elapsed=%dms",
        status, test_num, total, route_ok, audit_result["score"], elapsed_ms,
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
    total     = len(records)
    passed    = sum(1 for r in records if r["verdict"] == "pass")
    failed    = total - passed
    avg_score = sum(r["score"] for r in records) / total if total else 0.0
    route_mismatches = [r for r in records if not r["route_match"]]
    fail_records     = [r for r in records if r["verdict"] == "fail"]

    lines = [
        f"✅ Прошло: {passed}/{total}   "
        f"❌ Упало: {failed}/{total}   "
        f"⭐ Ср. скор: {avg_score:.2f}",
    ]

    if route_mismatches:
        lines.append("\n⚠️  Неверный route:")
        for r in route_mismatches:
            lines.append(f"  - [{r['expected_route']} -> {r['actual_route']}] {r['query'][:70]}")

    if fail_records:
        lines.append("\n❌  Проваленные тесты:")
        for r in fail_records[:5]:
            lines.append(f"  {r['query'][:70]}")
            for issue in r["issues"][:2]:
                lines.append(f"    - {issue}")

    return "\n".join(lines)


def run(query: str, history: list[dict] | None = None) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    explicit_n = _parse_n_from_query(query)
    n = explicit_n if explicit_n and explicit_n > 0 else len(ALL_ROUTES)
    n = min(n, 50)

    logger.info("[SelfTest] Starting run_id=%s  n=%d tests", run_id, n)
    report_progress(f"🧠 Самотест: генерирую {n} тестовых запросов...")

    try:
        cases = _generate_test_cases(n)
    except Exception as e:
        logger.error("[SelfTest] Failed to generate test cases: %s", e)
        return f"Не удалось сгенерировать тестовые запросы: {e}"

    if not cases:
        return "Ошибка: LLM не вернул ни одного тестового запроса."

    report_progress(f"🚀 Запускаю {len(cases)} тестов...")

    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        record = _run_single_test(case, test_num=i, total=len(cases))
        records.append(record)

    log_path = _save_log(records, run_id)
    summary  = _build_summary(records)

    return (
        f"🧠 Самотестирование завершено [{run_id}]\n\n"
        + summary
        + f"\n\n📄 Лог: {log_path.resolve()}"
    )
