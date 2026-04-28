"""brain/agents/self_test_agent.py

Self-Test Agent.

Normal mode  (e.g. "запусти 7 тестов"):
  Run N tests, print summary.

Self-heal mode (trigger words: фикс / исправь / авто / self-heal OR N >= 50):
  1. Run N tests (batched by 50)
  2. Collect FAIL records
  3. LLM analyses failures → bug list
  4. LLM fixes each affected file
  5. Push to new branch fix/self-heal-<run_id> via PyGithub or git CLI
  6. Return full summary + branch URL
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.client import chat, MODEL_HEAVY
from brain.ask import report_progress

logger = logging.getLogger(__name__)

LOGS_DIR    = Path("logs")
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
BRANCH_BASE = "fix/self-heal"
GITHUB_REPO = "s4s4s4s/Jarvis"
BASE_BRANCH = "feature/planner-agent"

ALL_ROUTES = ["chat", "code", "plan", "web", "tool", "memory", "deep"]

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

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
    "CRITICAL RULES for self-contained queries:\n"
    "- Queries must be in Russian (that is the user language).\n"
    "- Each query must be a complete, natural sentence, not a category label.\n"
    "- Cover ALL routes listed above at least once.\n"
    "- IMPORTANT: Every query must be SELF-CONTAINED — it must include all information\n"
    "  needed to answer it. For example:\n"
    "    BAD:  'Запусти этот скрипт'  (no script provided)\n"
    "    GOOD: 'Напиши скрипт на Python, который выводит числа Фибоначчи до 100'\n"
    "    BAD:  'Что я говорил вчера?'  (no context)\n"
    "    GOOD: 'Какую погоду ты мне показывал в последний раз?'\n"
    "- For route 'code': ask to WRITE or FIX code, never to run code without providing it.\n"
    "- For route 'memory': ask to recall facts Jarvis would plausibly know about the user.\n"
    "- Return ONLY a valid JSON array, no markdown, no extra text.\n"
    "\n"
    "Each element: {\"query\": \"<natural Russian sentence>\", \"expected_route\": \"<route>\"}"
)

_AUDIT_SYSTEM = (
    "You are a QA auditor evaluating an AI assistant called Jarvis.\n"
    "You receive the original user query, the route Jarvis chose, and Jarvis full response.\n"
    "\n"
    "Scoring guidelines:\n"
    "  1.0 — perfect: correct route, complete and accurate answer\n"
    "  0.8 — good: correct route, minor gaps or style issues\n"
    "  0.6 — acceptable: mostly correct, some missing detail\n"
    "  0.4 — poor: wrong route OR substantially incomplete answer\n"
    "  0.2 — bad: wrong route AND useless or harmful response\n"
    "  0.0 — critical failure: error, crash, hallucination of facts\n"
    "\n"
    "Verdict rule: 'pass' if score >= 0.5, 'fail' if score < 0.5.\n"
    "Be fair — a useful, mostly-correct response at the right route deserves >= 0.6.\n"
    "\n"
    "Respond with ONLY a valid JSON object, no markdown:\n"
    "{\"verdict\": \"pass\" | \"fail\", \"score\": <0.0-1.0>, "
    "\"issues\": [...], \"suggestions\": [...]}"
)

_ANALYST_SYSTEM = """\
You are a senior Python engineer doing post-mortem analysis of an AI assistant called Jarvis.

You receive a list of FAILED test records. Each record has:
  - query          : the user message
  - expected_route : the route Jarvis should have chosen
  - actual_route   : the route Jarvis actually chose
  - response_head  : first 400 chars of Jarvis response
  - issues         : list of issues found by the QA auditor
  - suggestions    : list of suggestions from the QA auditor

Your task: produce a concise, actionable bug report.

Rules:
  - Group related failures into one bug. Max 8 bugs.
  - Skip failures that are clearly test-generator errors
    (e.g. expected_route is obviously wrong for the query content).
  - Only include bugs in files that actually exist in brain/ or tools/.

Respond ONLY with a valid JSON object (no markdown):
{
  "bugs": [
    {
      "id": "BUG-1",
      "title": "short title",
      "severity": "high|medium|low",
      "affected_files": ["brain/agents/foo.py"],
      "root_cause": "one sentence",
      "fix_description": "concrete instructions for the fix",
      "test_ids": [0, 1, 2]
    }
  ]
}
"""

_FIXER_SYSTEM = """\
You are a senior Python engineer fixing a bug in the Jarvis AI assistant codebase.

You receive:
  - The bug description (title, root cause, fix instructions)
  - The full current content of the file to fix

Return the COMPLETE fixed file content as a JSON object:
  {"fixed_content": "<entire fixed file as a string>"}

Rules:
  - Return ONLY valid JSON, no markdown, no extra text.
  - Preserve all existing functionality; only change what is needed.
  - Add a short comment # FIX <BUG_ID>: <title> near each changed section.
  - If the bug does not affect this file, return the original content unchanged.
"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Query parsing
# ─────────────────────────────────────────────────────────────────────────────

_SELF_HEAL_KEYWORDS = re.compile(
    r"(фикс|исправь|исправ|автоматически|авто|self[- ]heal|push|запуши|запуш|ветку|сам)",
    re.IGNORECASE,
)


def _parse_query(query: str) -> tuple[int, bool]:
    """Return (n_tests, self_heal_mode)."""
    m = re.search(r"(\d+)\s*тест", query)
    n = int(m.group(1)) if m else len(ALL_ROUTES)
    # self-heal if explicitly requested OR n >= 50
    heal = bool(_SELF_HEAL_KEYWORDS.search(query)) or n >= 50
    return n, heal


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — generate test cases
# ─────────────────────────────────────────────────────────────────────────────

def _generate_test_cases(n: int) -> list[dict]:
    prompt = (
        f"Generate exactly {n} test queries. "
        f"Cover ALL 7 routes at least once. "
        f"Every query must be self-contained."
    )
    messages = [
        {"role": "system", "content": _GENERATE_SYSTEM},
        {"role": "user",   "content": prompt},
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — run single test
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_test(case: dict, test_num: int, total: int) -> dict:
    query          = case["query"]
    expected_route = case["expected_route"]

    report_progress(f"⏳ Тест {test_num}/{total}: {query[:80]}")
    logger.info("[SelfTest] Test %d/%d - %s (expected=%s)", test_num, total, query[:70], expected_route)

    t0             = time.monotonic()
    actual_route   = "unknown"
    response       = ""
    pipeline_error: str | None = None

    try:
        from brain.ask import _route, _dispatch
        route_data   = _route(query, history=[])
        actual_route = route_data.get("route", "unknown")
        response     = _dispatch(route_data, query, history=[])
    except Exception as e:
        pipeline_error = str(e)
        response       = f"[PIPELINE ERROR] {e}"
        logger.error("[SelfTest] Pipeline error for '%s': %s", query[:60], e)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    audit_result: dict[str, Any] = {
        "verdict":     "fail",
        "score":       0.0,
        "issues":      [f"Pipeline error: {pipeline_error}"] if pipeline_error else [],
        "suggestions": [],
    }

    if not pipeline_error:
        try:
            audit_messages = [
                {"role": "system", "content": _AUDIT_SYSTEM},
                {"role": "user",   "content": (
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

    status   = "✅" if audit_result["verdict"] == "pass" else "❌"
    route_ok = "✔" if record["route_match"] else f"✘ got={actual_route}"
    report_progress(
        f"{status} {test_num}/{total}  route={route_ok}  "
        f"score={audit_result['score']:.2f}  {elapsed_ms // 1000}s  «{query[:55]}»"
    )
    return record


def _run_batch(n: int) -> list[dict]:
    """Generate and run one batch of up to 50 tests."""
    n = min(n, 50)
    report_progress(f"🧠 Генерирую {n} тестовых запросов...")
    try:
        cases = _generate_test_cases(n)
    except Exception as e:
        logger.error("[SelfTest] Failed to generate cases: %s", e)
        return []
    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        records.append(_run_single_test(case, test_num=i, total=len(cases)))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — analyse failures
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_failures(failures: list[dict]) -> list[dict]:
    if not failures:
        return []
    trimmed = [
        {
            "index":          i,
            "query":          r["query"],
            "expected_route": r["expected_route"],
            "actual_route":   r["actual_route"],
            "issues":         r["issues"],
            "suggestions":    r["suggestions"],
            "response_head":  r["response"][:400],
        }
        for i, r in enumerate(failures)
    ]
    prompt = (
        f"Here are {len(trimmed)} failed tests. Analyse and produce the bug report.\n\n"
        + json.dumps(trimmed, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": _ANALYST_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    raw = chat(model=MODEL_HEAVY, messages=messages, options={"temperature": 0.1, "num_ctx": 16384})
    try:
        data = _extract_json_object(raw)
        bugs: list[dict] = data.get("bugs", [])
    except Exception as e:
        logger.error("[SelfTest] Bug analysis parse error: %s", e)
        bugs = []
    logger.info("[SelfTest] Identified %d bugs", len(bugs))
    return bugs


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — auto-fix
# ─────────────────────────────────────────────────────────────────────────────

def _fix_bug(bug: dict, failures: list[dict]) -> dict[str, str]:
    """Returns {rel_path: fixed_content}."""
    results: dict[str, str] = {}
    for rel_path in bug.get("affected_files", []):
        path = REPO_ROOT / rel_path
        if not path.exists():
            logger.warning("[SelfTest] File not found: %s", path)
            continue
        current = path.read_text(encoding="utf-8")

        rel_failures = [failures[i] for i in bug.get("test_ids", []) if i < len(failures)]
        failure_ctx  = json.dumps(
            [{"query": f["query"], "response_head": f["response"][:300]} for f in rel_failures],
            ensure_ascii=False,
        )
        prompt = textwrap.dedent(f"""\
            Bug ID:           {bug['id']}
            Title:            {bug['title']}
            Root cause:       {bug['root_cause']}
            Fix instructions: {bug['fix_description']}

            Relevant failing queries:
            {failure_ctx}

            File to fix: {rel_path}
            ===BEGIN FILE===
            {current}
            ===END FILE===
        """)
        messages = [
            {"role": "system", "content": _FIXER_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        raw = chat(model=MODEL_HEAVY, messages=messages,
                   options={"temperature": 0.05, "num_ctx": 32768})
        try:
            data  = _extract_json_object(raw)
            fixed = data.get("fixed_content", "")
            if fixed:
                results[rel_path] = fixed
                logger.info("[SelfTest] Fixed %s (%d chars)", rel_path, len(fixed))
        except Exception as e:
            logger.error("[SelfTest] Fixer parse error for %s: %s", rel_path, e)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — push to new GitHub branch
# ─────────────────────────────────────────────────────────────────────────────

def _push_to_github(branch: str, files: dict[str, str], commit_msg: str) -> bool:
    """Try PyGithub first, fall back to git CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        try:
            return _push_pygithub(token, branch, files, commit_msg)
        except Exception as e:
            logger.warning("[SelfTest] PyGithub push failed: %s — falling back to git CLI", e)
    return _push_git_cli(branch, files, commit_msg)


def _push_pygithub(token: str, branch: str, files: dict[str, str], commit_msg: str) -> bool:
    from github import Github  # type: ignore
    g          = Github(token)
    repo       = g.get_repo(GITHUB_REPO)
    base_sha   = repo.get_branch(BASE_BRANCH).commit.sha
    repo.create_git_ref(ref=f"refs/heads/{branch}", sha=base_sha)
    for rel_path, content in files.items():
        try:
            existing = repo.get_contents(rel_path, ref=branch)
            repo.update_file(rel_path, commit_msg, content, existing.sha, branch=branch)
        except Exception:
            repo.create_file(rel_path, commit_msg, content, branch=branch)
        logger.info("[SelfTest] PyGithub pushed %s", rel_path)
    return True


def _push_git_cli(branch: str, files: dict[str, str], commit_msg: str) -> bool:
    try:
        subprocess.run(["git", "checkout", "-b", branch], cwd=REPO_ROOT, check=True)
        for rel_path, content in files.items():
            full = REPO_ROOT / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            subprocess.run(["git", "add", rel_path], cwd=REPO_ROOT, check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
        subprocess.run(["git", "push", "-u", "origin", branch], cwd=REPO_ROOT, check=True)
        logger.info("[SelfTest] git CLI pushed branch '%s'", branch)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[SelfTest] git CLI push failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_log(records: list[dict], run_id: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"self_test_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return log_path


def _build_summary(records: list[dict]) -> str:
    total    = len(records)
    passed   = sum(1 for r in records if r["verdict"] == "pass")
    failed   = total - passed
    avg      = sum(r["score"] for r in records) / total if total else 0.0
    mismatches  = [r for r in records if not r["route_match"]]
    fail_recs   = [r for r in records if r["verdict"] == "fail"]

    lines = [
        f"✅ Прошло: {passed}/{total}   "
        f"❌ Упало: {failed}/{total}   "
        f"⭐ Ср. скор: {avg:.2f}",
    ]
    if mismatches:
        lines.append("\n⚠️  Неверный route:")
        for r in mismatches:
            lines.append(f"  - [{r['expected_route']} → {r['actual_route']}] {r['query'][:70]}")
    if fail_recs:
        lines.append("\n❌  Проваленные тесты:")
        for r in fail_recs[:5]:
            lines.append(f"  {r['query'][:70]}")
            for issue in r["issues"][:2]:
                lines.append(f"    - {issue}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(query: str, history: list[dict] | None = None) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n, self_heal = _parse_query(query)
    n = max(1, min(n, 200))  # hard cap 200

    logger.info("[SelfTest] run_id=%s  n=%d  self_heal=%s", run_id, n, self_heal)

    # ── run tests in batches of 50 ────────────────────────────────────────────
    all_records: list[dict] = []
    remaining = n
    batch_num = 0
    while remaining > 0:
        batch_num += 1
        size = min(remaining, 50)
        report_progress(f"🚀 Батч {batch_num}: запускаю {size} тестов...")
        all_records.extend(_run_batch(size))
        remaining -= size

    log_path = _save_log(all_records, run_id)
    summary  = _build_summary(all_records)

    base_reply = (
        f"🧠 Самотестирование завершено [{run_id}]\n\n"
        + summary
        + f"\n\n📄 Лог: {log_path.resolve()}"
    )

    if not self_heal:
        return base_reply

    # ── self-heal pipeline ────────────────────────────────────────────────────
    failures = [r for r in all_records if r["verdict"] == "fail"]
    if not failures:
        return base_reply + "\n\n✅ Фейлов нет — фиксить нечего."

    report_progress(f"🔍 Анализирую {len(failures)} фейлов...")
    bugs = _analyse_failures(failures)

    # save bug report
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    bug_report_path = LOGS_DIR / f"self_heal_bugs_{run_id}.json"
    bug_report_path.write_text(
        json.dumps({"run_id": run_id, "bugs": bugs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not bugs:
        return base_reply + "\n\n⚠️ LLM не нашёл акционируемых багов (возможно, фейлы в генераторе тестов)."

    bug_lines = []
    for b in bugs:
        bug_lines.append(f"  [{b['id']}] [{b['severity'].upper()}] {b['title']}")
        bug_lines.append(f"         {b['root_cause']}")

    report_progress(f"🛠 Фиксирую {len(bugs)} багов...")
    all_fixed: dict[str, str] = {}
    for bug in bugs:
        if bug.get("severity") == "low" and len(bugs) > 4:
            continue  # skip low-severity when there are bigger issues
        report_progress(f"🛠 Фиксю {bug['id']}: {bug['title']}...")
        fixed = _fix_bug(bug, failures)
        all_fixed.update(fixed)

    if not all_fixed:
        return (
            base_reply
            + "\n\n🐛 Баги:\n" + "\n".join(bug_lines)
            + "\n\n❌ Не удалось автоматически сгенерировать фиксы."
        )

    new_branch  = f"{BRANCH_BASE}-{run_id}"
    bug_ids     = ", ".join(b["id"] for b in bugs)
    commit_msg  = f"fix: self-heal [{run_id}] auto-fix {bug_ids}"

    report_progress(f"🚀 Пушу в ветку {new_branch}...")
    ok = _push_to_github(new_branch, all_fixed, commit_msg)

    push_status = (
        f"✅ Пуш успешен!\n"
        f"   Ветка: https://github.com/{GITHUB_REPO}/tree/{new_branch}\n"
        f"   Коммит: {commit_msg}"
        if ok else
        f"❌ Пуш не удался. Файлы записаны локально.\n"
        f"   git checkout -b {new_branch} && git add -A && "
        f"git commit -m '{commit_msg}' && git push origin {new_branch}"
    )

    return (
        base_reply
        + f"\n\n🐛 Баги ({len(bugs)} шт.):\n" + "\n".join(bug_lines)
        + f"\n\n🛠 Исправлено файлов: {list(all_fixed.keys())}\n\n"
        + push_status
    )
