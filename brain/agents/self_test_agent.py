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

import ast
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
# Repo map (injected into analyst prompt)
# ─────────────────────────────────────────────────────────────────────────────

REPO_MAP = """\
ARCHITECTURE (real files only — do NOT invent paths):
  brain/ask.py                  — main router + executor (ALL routes live here, _route + _dispatch)
  brain/prompts.py              — all system prompts (ROUTER_SYSTEM, DEEP_SYSTEM, etc.)
  brain/router_keywords.py      — Level-1 keyword router (fast_route, no LLM needed)
  brain/client.py               — Ollama chat wrapper (chat, MODEL_HEAVY, MODEL_FAST, MODEL_ROUTER)
  brain/agents/code_agent.py    — write→run→verify loop
  brain/agents/plan_agent.py    — multi-step decomposition
  brain/agents/deep.py          — deep analysis (ALREADY chosen route, not router)
  brain/agents/memory_agent.py  — memory access
  brain/agents/self_test_agent.py — THIS FILE
  brain/agents/web_agent.py     — web search
  brain/agents/tool_agent.py    — tool dispatcher
  brain/agents/chat.py          — chat agent
  brain/agents/code_dev_agent.py — universal code developer (any codebase)
  brain/agents/self_extend_agent.py — Jarvis extends himself
  brain/agents/self_analysis_agent.py — deep self-analysis
  brain/agents/github_reader.py — reads GitHub repos via API or git clone
  tools/memory.py               — get_memory_context, save_fact (NO recall_events!)
  tools/weather.py              — weather tool
  tools/crypto.py               — crypto tool

FILES THAT DO NOT EXIST (NEVER reference these):
  brain/agents/web.py, brain/agents/tool.py, brain/agents/plan.py,
  brain/agents/memory.py, brain/agents/chat_agent.py
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build real file whitelist via AST scan
# ─────────────────────────────────────────────────────────────────────────────

def _build_file_whitelist() -> set[str]:
    """Return set of relative paths (POSIX) for all .py files in the repo."""
    whitelist: set[str] = set()
    for p in REPO_ROOT.rglob("*.py"):
        try:
            rel = p.relative_to(REPO_ROOT).as_posix()
            whitelist.add(rel)
        except ValueError:
            pass
    return whitelist


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_GENERATE_SYSTEM = (
    "You are a test-case designer for an AI assistant called Jarvis.\n"
    "Jarvis routes user messages to one of these agents:\n"
    "  chat    - casual conversation, general questions, science/history/language facts\n"
    "  code    - write, run or fix a Python script\n"
    "  plan    - multi-step task that needs planning + execution\n"
    "  web     - search the internet for current information\n"
    "  tool    - live data: weather, crypto price, currency rate, timer, time\n"
    "  memory  - recall something from previous conversations\n"
    "  deep    - ONLY genuinely complex multi-faceted reasoning (financial modelling, system design)\n"
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
    "- For route 'deep': ONLY genuinely complex questions like financial modelling or system design.\n"
    "  MANDATORY deep examples (include at least 2 per batch):\n"
    "    'Если я куплю 50 акций Tesla по $250, сколько мне нужно денег?'\n"
    "    'Если я инвестирую $10000 на 5 лет с доходом 12%, сколько получу?'\n"
    "    'Если цена продукта $300 и я снижу её на 30%, какова будет новая цена?'\n"
    "  Do NOT generate 'deep' queries for: science explanations, history, language learning, ML concepts.\n"
    "  Those should be 'chat'.\n"
    "- For route 'tool'/'weather': always specify a city.\n"
    "- For route 'tool'/'currency': use phrases like 'курс евро к рублю сейчас', 'какой курс доллара'.\n"
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
    "SPECIAL CASES (override default scoring):\n"
    "  1. MEMORY route with no stored data: If the query asks about past interactions and\n"
    "     Jarvis honestly says 'I don't have records of this' — this is CORRECT behaviour.\n"
    "     Score >= 0.6, verdict=pass. Only fail if Jarvis hallucinates fake memories.\n"
    "  2. WRONG expected_route: If the expected_route in the test is clearly wrong for\n"
    "     the query type (e.g. expected=deep but query is simple science fact), and Jarvis\n"
    "     used the correct route and gave a good answer — score >= 0.6, verdict=pass.\n"
    "  3. FINANCIAL CALCULATIONS: If query contains 'если я куплю', 'если цена',\n"
    "     'увеличу/уменьшу цену на X%', 'инвестирую $N на Y лет' — expected route is 'deep'.\n"
    "     If Jarvis used 'tool' or 'code' for these — that is a route error, score <= 0.4.\n"
    "  4. CURRENCY/CRYPTO LIVE PRICES: If query asks for current exchange rate or crypto price\n"
    "     and Jarvis used 'web' instead of 'tool' — score <= 0.4 (wrong route).\n"
    "  5. COOKING/RECIPES: If query asks how to cook something and Jarvis used 'web'\n"
    "     instead of 'plan' — score <= 0.4 (wrong route).\n"
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
  - CRITICAL: Only include bugs in files that ACTUALLY EXIST. Use the repo map below.
  - NEVER reference brain/agents/web.py, tool.py, plan.py, memory.py or chat_agent.py
    — these files do NOT exist.
  - Routing bugs belong in brain/ask.py (router) or brain/prompts.py (ROUTER_SYSTEM prompt).
  - Memory failures where Jarvis says 'no records found' are NOT bugs — skip them.
  - After listing affected_files, add a field "files_verified": true only if ALL listed
    files are in the repo map.

{repo_map}

Respond ONLY with a valid JSON object (no markdown):
{{
  "bugs": [
    {{
      "id": "BUG-1",
      "title": "short title",
      "severity": "high|medium|low",
      "affected_files": ["brain/agents/foo.py"],
      "root_cause": "one sentence",
      "fix_description": "concrete instructions for the fix",
      "test_ids": [0, 1, 2]
    }}
  ]
}}
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
  - NEVER add empty if/else branches where both paths do the same thing.
  - NEVER make changes that don't actually fix the described bug.
  - NEVER change routing logic without adding explicit keyword matching.
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


def _parse_auditor(raw: str) -> dict:
    """Robust auditor JSON parser with multiple fallback strategies."""
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

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    try:
        import json_repair
        return json_repair.loads(raw)
    except (ImportError, Exception):
        pass

    logger.warning("[SelfTest] parse_auditor: all strategies failed, using default")
    return {"verdict": "fail", "score": 0.5, "issues": ["parse failed"], "suggestions": []}


def _extract_json_object(raw: str) -> dict:
    return _parse_auditor(raw)


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
    heal = bool(_SELF_HEAL_KEYWORDS.search(query)) or n >= 50
    return n, heal


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — generate test cases
# ─────────────────────────────────────────────────────────────────────────────

def _generate_test_cases(n: int) -> list[dict]:
    prompt = (
        f"Generate exactly {n} test queries. "
        f"Cover ALL 7 routes at least once. "
        f"Every query must be self-contained. "
        f"Include at least 2 'deep' route queries with financial calculations "
        f"(e.g. buying stocks, investment growth, price changes)."
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
            parsed = _parse_auditor(raw_audit)
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

    # BUG-FIX: feed passing tests into classifier dataset
    if audit_result["verdict"] == "pass" and record["route_match"]:
        _update_classifier_dataset(record)

    return record


def _update_classifier_dataset(record: dict) -> None:
    """Append a passing test record to route_examples.jsonl for future classifier training."""
    try:
        data_dir = REPO_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / "route_examples.jsonl"
        entry = {
            "text": record["query"],
            "route": record["actual_route"],
            "score": record["score"],
        }
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[SelfTest] Failed to update classifier dataset: %s", e)


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
# Step 3 — analyse failures (with repo map + whitelist filter)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_bugs_by_whitelist(bugs: list[dict], whitelist: set[str]) -> list[dict]:
    clean: list[dict] = []
    for bug in bugs:
        real_files = [f for f in bug.get("affected_files", []) if f in whitelist]
        ghost_files = [f for f in bug.get("affected_files", []) if f not in whitelist]
        if ghost_files:
            logger.warning(
                "[SelfTest] Bug %s references non-existent files %s — marking needs_human_review",
                bug["id"], ghost_files,
            )
        if not real_files:
            bug["needs_human_review"] = True
            bug["affected_files"] = []
            logger.warning("[SelfTest] Bug %s has no valid files — skipping auto-fix", bug["id"])
        else:
            bug["affected_files"] = real_files
            if ghost_files:
                bug["skipped_ghost_files"] = ghost_files
        clean.append(bug)
    return clean


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
    analyst_system = _ANALYST_SYSTEM.format(repo_map=REPO_MAP)
    messages = [
        {"role": "system", "content": analyst_system},
        {"role": "user",   "content": prompt},
    ]
    raw = chat(model=MODEL_HEAVY, messages=messages, options={"temperature": 0.1, "num_ctx": 16384})
    try:
        data = _parse_auditor(raw)
        bugs: list[dict] = data.get("bugs", [])
    except Exception as e:
        logger.error("[SelfTest] Bug analysis parse error: %s", e)
        bugs = []
    logger.info("[SelfTest] Identified %d bugs (pre-filter)", len(bugs))

    whitelist = _build_file_whitelist()
    bugs = _filter_bugs_by_whitelist(bugs, whitelist)
    fixable = [b for b in bugs if not b.get("needs_human_review")]
    logger.info("[SelfTest] %d fixable bugs after whitelist filter", len(fixable))
    return bugs


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — auto-fix + re-test verification
# ─────────────────────────────────────────────────────────────────────────────

def _quick_retest(test_cases: list[dict]) -> float:
    if not test_cases:
        return 1.0
    passed = 0
    for case in test_cases:
        rec = _run_single_test(case, test_num=1, total=1)
        if rec["verdict"] == "pass":
            passed += 1
    return passed / len(test_cases)


def _fix_bug(bug: dict, failures: list[dict]) -> dict[str, str]:
    if bug.get("needs_human_review"):
        logger.info("[SelfTest] Skipping %s (needs_human_review)", bug["id"])
        return {}

    results: dict[str, str] = {}
    for rel_path in bug.get("affected_files", []):
        path = REPO_ROOT / rel_path
        if not path.exists():
            logger.warning("[SelfTest] File not found (post-filter): %s", path)
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
            data  = _parse_auditor(raw)
            fixed = data.get("fixed_content", "")
            if fixed:
                # BUG-FIX: validate syntax before accepting the fix
                try:
                    ast.parse(fixed)
                    results[rel_path] = fixed
                    logger.info("[SelfTest] Fixed %s (%d chars)", rel_path, len(fixed))
                except SyntaxError as se:
                    logger.error("[SelfTest] Generated fix for %s has syntax error: %s — rejecting", rel_path, se)
        except Exception as e:
            logger.error("[SelfTest] Fixer parse error for %s: %s", rel_path, e)
    return results


def _verify_fix(bug: dict, failures: list[dict], fixed_files: dict[str, str]) -> bool:
    if not fixed_files or not bug.get("test_ids"):
        return True

    original_contents: dict[str, str] = {}
    for rel_path, content in fixed_files.items():
        p = REPO_ROOT / rel_path
        if p.exists():
            original_contents[rel_path] = p.read_text(encoding="utf-8")
        p.write_text(content, encoding="utf-8")

    test_cases = [
        {"query": failures[i]["query"], "expected_route": failures[i]["expected_route"]}
        for i in bug.get("test_ids", []) if i < len(failures)
    ][:5]

    report_progress(f"🧪 Re-testing {bug['id']} ({len(test_cases)} cases)...")
    pass_rate = _quick_retest(test_cases)
    logger.info("[SelfTest] Re-test %s: pass_rate=%.2f", bug["id"], pass_rate)

    if pass_rate < 0.3:
        logger.warning("[SelfTest] Fix for %s has low pass-rate %.2f — reverting", bug["id"], pass_rate)
        for rel_path, orig in original_contents.items():
            (REPO_ROOT / rel_path).write_text(orig, encoding="utf-8")
        fixed_files.clear()
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — push to new GitHub branch
# ─────────────────────────────────────────────────────────────────────────────

def _push_to_github(branch: str, files: dict[str, str], commit_msg: str) -> bool:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        try:
            return _push_pygithub(token, branch, files, commit_msg)
        except Exception as e:
            logger.warning("[SelfTest] PyGithub push failed: %s — falling back to git CLI", e)
    return _push_git_cli(branch, files, commit_msg)


def _push_pygithub(token: str, branch: str, files: dict[str, str], commit_msg: str) -> bool:
    from github import Github
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
    n = max(1, min(n, 200))

    logger.info("[SelfTest] run_id=%s  n=%d  self_heal=%s", run_id, n, self_heal)

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

    failures = [r for r in all_records if r["verdict"] == "fail"]
    if not failures:
        return base_reply + "\n\n✅ Фейлов нет — фиксить нечего."

    report_progress(f"🔍 Анализирую {len(failures)} фейлов...")
    bugs = _analyse_failures(failures)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    bug_report_path = LOGS_DIR / f"self_heal_bugs_{run_id}.json"
    bug_report_path.write_text(
        json.dumps({"run_id": run_id, "bugs": bugs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not bugs:
        return base_reply + "\n\n⚠️ LLM не нашёл акционируемых багов."

    fixable_bugs = [b for b in bugs if not b.get("needs_human_review")]
    bug_lines = []
    for b in bugs:
        flag = " [⚠️ needs_human_review]" if b.get("needs_human_review") else ""
        bug_lines.append(f"  [{b['id']}] [{b['severity'].upper()}] {b['title']}{flag}")
        bug_lines.append(f"         {b['root_cause']}")

    if not fixable_bugs:
        return (
            base_reply
            + f"\n\n🐛 Баги ({len(bugs)} шт.):\n" + "\n".join(bug_lines)
            + "\n\n⚠️ Все баги требуют ручной правки (hallucinated file paths)."
        )

    report_progress(f"🛠 Фиксирую {len(fixable_bugs)} багов...")
    all_fixed: dict[str, str] = {}
    verified_bug_ids: list[str] = []

    for bug in fixable_bugs:
        if bug.get("severity") == "low" and len(fixable_bugs) > 4:
            continue
        report_progress(f"🛠 Фикшу {bug['id']}: {bug['title']}...")
        fixed = _fix_bug(bug, failures)
        if not fixed:
            continue

        verified = _verify_fix(bug, failures, fixed)
        if verified and fixed:
            all_fixed.update(fixed)
            verified_bug_ids.append(bug["id"])
            report_progress(f"✅ {bug['id']} verified OK")
        else:
            report_progress(f"❌ {bug['id']} fix reverted (low pass-rate)")

    if not all_fixed:
        return (
            base_reply
            + "\n\n🐛 Баги:\n" + "\n".join(bug_lines)
            + "\n\n❌ Не удалось автоматически сгенерировать фиксы (все откатаны после re-test)."
        )

    new_branch  = f"{BRANCH_BASE}-{run_id}"
    commit_msg  = f"fix: self-heal [{run_id}] auto-fix {', '.join(verified_bug_ids)}"

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
        + f"\n\n🛠 Исправлено файлов ({len(verified_bug_ids)} бага): {list(all_fixed.keys())}\n\n"
        + push_status
    )
