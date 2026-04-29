"""scripts/self_heal.py

Autonomous self-healing pipeline for Jarvis.

Steps:
  1. Run self-test with N questions (default 100, capped at 50 per batch due to self_test_agent cap)
     by running two batches of 50.
  2. Collect all FAIL records from the log.
  3. Ask LLM to analyse failures and produce a prioritised bug list.
  4. For each bug: read the affected file, ask LLM to produce a fixed version.
  5. Create a new git branch  fix/self-heal-<timestamp>  and push all fixed files.
  6. Print a summary with the new branch name.

Usage (from repo root):
  python scripts/self_heal.py          # 100 tests (2 x 50)
  python scripts/self_heal.py --n 50   # single batch of 50
  python scripts/self_heal.py --n 10   # quick smoke run

Requirements:
  - GITHUB_TOKEN env var OR the token is already configured for git push.
  - pip install PyGithub  (only needed for push; falls back to git CLI)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- repo root on sys.path so we can import brain.* ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from brain.client import chat, MODEL_HEAVY  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("self_heal")

LOGS_DIR   = REPO_ROOT / "logs"
BRANCH_BASE = "fix/self-heal"

# ---------------------------------------------------------------------------
# Step 1 — run self-test
# ---------------------------------------------------------------------------

def _run_self_test_batch(n: int) -> list[dict]:
    """Run one batch of up to 50 tests via self_test_agent.run().
    Returns list of test records parsed from the saved log file.
    """
    from brain.agents.self_test_agent import run as st_run

    logger.info("[self_heal] Running self-test batch: n=%d", n)
    summary_text = st_run(query=f"запусти {n} тестов")
    logger.info("[self_heal] Batch summary:\n%s", summary_text)

    # Find the latest log written during this batch
    log_files = sorted(LOGS_DIR.glob("self_test_*.json"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        logger.warning("[self_heal] No log files found after batch run")
        return []

    latest = log_files[-1]
    logger.info("[self_heal] Reading log: %s", latest)
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def collect_records(total_n: int) -> list[dict]:
    """Run ceil(total_n / 50) batches of up to 50 tests each."""
    BATCH = 50
    all_records: list[dict] = []
    remaining = total_n
    batch_num  = 0
    while remaining > 0:
        batch_num += 1
        size = min(remaining, BATCH)
        logger.info("[self_heal] Batch %d: %d tests", batch_num, size)
        records = _run_self_test_batch(size)
        all_records.extend(records)
        remaining -= size
    return all_records


# ---------------------------------------------------------------------------
# Step 2 — extract failures
# ---------------------------------------------------------------------------

def extract_failures(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("verdict") == "fail"]


# ---------------------------------------------------------------------------
# Step 3 — bug analysis
# ---------------------------------------------------------------------------

_ANALYST_SYSTEM = """\
You are a senior Python engineer doing post-mortem analysis of an AI assistant called Jarvis.

You receive a list of FAILED test records. Each record has:
  - query          : the user message
  - expected_route : the route Jarvis should have chosen
  - actual_route   : the route Jarvis actually chose
  - response       : Jarvis full response
  - issues         : list of issues found by the QA auditor
  - suggestions    : list of suggestions from the QA auditor

Your task: produce a concise, actionable bug report.

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
      "test_ids": [0, 1, 2]   // indices into the failures array
    }
  ]
}

Group related failures into one bug. Max 10 bugs. Skip failures that are clearly test-generator
errors (e.g. expected_route is wrong for the query content).
"""


def analyse_failures(failures: list[dict]) -> list[dict]:
    if not failures:
        logger.info("[self_heal] No failures — nothing to analyse")
        return []

    # Trim responses so the prompt fits in context
    trimmed = []
    for i, r in enumerate(failures):
        trimmed.append({
            "index":          i,
            "query":          r["query"],
            "expected_route": r["expected_route"],
            "actual_route":   r["actual_route"],
            "issues":         r["issues"],
            "suggestions":    r["suggestions"],
            "response_head":  r["response"][:400],
        })

    prompt = (
        f"Here are {len(trimmed)} failed tests. Analyse and produce the bug report.\n\n"
        + json.dumps(trimmed, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": _ANALYST_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    raw = chat(model=MODEL_HEAVY, messages=messages, options={"temperature": 0.1, "num_ctx": 16384})

    raw = raw.strip()
    # strip markdown fences if present
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

    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        logger.error("[self_heal] Bug analysis returned no JSON: %s", raw[:300])
        return []

    data = json.loads(raw[start:end + 1])
    bugs: list[dict] = data.get("bugs", [])
    logger.info("[self_heal] Identified %d bugs", len(bugs))
    return bugs


# ---------------------------------------------------------------------------
# Step 4 — auto-fix
# ---------------------------------------------------------------------------

_FIXER_SYSTEM = """\
You are a senior Python engineer fixing a bug in the Jarvis AI assistant codebase.

You receive:
  - The bug description (title, root cause, fix instructions)
  - The full current content of the file to fix

Your task: return the COMPLETE fixed file content as a JSON object:
  {"fixed_content": "<entire fixed file as a string>"}

Rules:
  - Return ONLY valid JSON, no markdown, no extra text.
  - Preserve all existing functionality; only change what's needed for the fix.
  - Add a short comment # FIX <BUG_ID>: <title> near each changed line.
  - If the fix requires changes to multiple files, fix only the file provided;
    the caller will invoke you separately for each file.
"""


def _read_file(rel_path: str) -> str | None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        logger.warning("[self_heal] File not found: %s", path)
        return None
    return path.read_text(encoding="utf-8")


def fix_bug(bug: dict, failures: list[dict]) -> dict[str, str]:
    """Returns {rel_path: fixed_content} for each affected file."""
    results: dict[str, str] = {}

    for rel_path in bug.get("affected_files", []):
        current = _read_file(rel_path)
        if current is None:
            continue

        relevant_failures = [
            failures[i] for i in bug.get("test_ids", []) if i < len(failures)
        ]
        failure_ctx = json.dumps(
            [{"query": f["query"], "response_head": f["response"][:300]} for f in relevant_failures],
            ensure_ascii=False,
        )

        prompt = textwrap.dedent(f"""\
            Bug ID:        {bug['id']}
            Title:         {bug['title']}
            Root cause:    {bug['root_cause']}
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
        raw = chat(
            model=MODEL_HEAVY,
            messages=messages,
            options={"temperature": 0.05, "num_ctx": 32768},
        )

        # parse JSON
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

        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1:
            logger.error("[self_heal] Fixer returned no JSON for %s: %s", rel_path, raw[:200])
            continue

        data = json.loads(raw[s:e + 1])
        fixed = data.get("fixed_content")
        if not fixed:
            logger.warning("[self_heal] Empty fixed_content for %s", rel_path)
            continue

        results[rel_path] = fixed
        logger.info("[self_heal] Fixed %s (%d chars)", rel_path, len(fixed))

    return results


# ---------------------------------------------------------------------------
# Step 5 — git push to new branch via GitHub API
# ---------------------------------------------------------------------------

def _github_push(branch: str, files: dict[str, str], commit_msg: str) -> bool:
    """Push multiple files to GitHub on a new branch using gh CLI or PyGithub."""

    # Try gh CLI first (simplest)
    gh = _find_gh_cli()
    if gh:
        return _push_via_gh_cli(gh, branch, files, commit_msg)

    # Fallback: PyGithub
    try:
        return _push_via_pygithub(branch, files, commit_msg)
    except ImportError:
        logger.error("[self_heal] Neither 'gh' CLI nor PyGithub found. Install: pip install PyGithub")
        return False


def _find_gh_cli() -> str | None:
    for candidate in ("gh", "gh.exe"):
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _push_via_gh_cli(gh: str, branch: str, files: dict[str, str], commit_msg: str) -> bool:
    """Create branch from current HEAD, write files, commit, push using git + gh."""
    try:
        # Create and checkout new branch locally
        subprocess.run(["git", "checkout", "-b", branch], cwd=REPO_ROOT, check=True)

        # Write files
        for rel_path, content in files.items():
            full = REPO_ROOT / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            subprocess.run(["git", "add", rel_path], cwd=REPO_ROOT, check=True)

        # Commit
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)

        # Push via gh (handles auth)
        subprocess.run(
            [gh, "repo", "sync", "--force"],
            cwd=REPO_ROOT,
            check=False,  # non-critical
        )
        subprocess.run(["git", "push", "-u", "origin", branch], cwd=REPO_ROOT, check=True)

        logger.info("[self_heal] Pushed branch '%s' via gh CLI", branch)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[self_heal] gh CLI push failed: %s", e)
        return False


def _push_via_pygithub(branch: str, files: dict[str, str], commit_msg: str) -> bool:
    from github import Github  # type: ignore

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("[self_heal] GITHUB_TOKEN env var not set")
        return False

    g    = Github(token)
    repo = g.get_repo("s4s4s4s/Jarvis")

    # Get current HEAD of feature/planner-agent
    base_branch = repo.get_branch("feature/planner-agent")
    base_sha    = base_branch.commit.sha

    # Create new branch
    repo.create_git_ref(ref=f"refs/heads/{branch}", sha=base_sha)
    logger.info("[self_heal] Created branch '%s' from %s", branch, base_sha[:8])

    # Push each file
    for rel_path, content in files.items():
        try:
            existing = repo.get_contents(rel_path, ref=branch)
            repo.update_file(
                path=rel_path,
                message=commit_msg,
                content=content,
                sha=existing.sha,
                branch=branch,
            )
        except Exception:
            repo.create_file(
                path=rel_path,
                message=commit_msg,
                content=content,
                branch=branch,
            )
        logger.info("[self_heal] Pushed %s to %s", rel_path, branch)

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_bug_report(bugs: list[dict], records: list[dict], run_id: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"self_heal_bugs_{run_id}.json"
    payload = {
        "run_id": run_id,
        "total_tests": len(records),
        "failures": len([r for r in records if r["verdict"] == "fail"]),
        "bugs": bugs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[self_heal] Bug report saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis self-healing pipeline")
    parser.add_argument("--n", type=int, default=100,
                        help="Total number of test questions (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse and print fixes but do NOT push to GitHub")
    parser.add_argument("--branch", type=str, default="",
                        help="Override target branch name")
    args = parser.parse_args()

    run_id    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    new_branch = args.branch or f"{BRANCH_BASE}-{run_id}"

    print(f"\n{'='*60}")
    print(f"  JARVIS SELF-HEAL  |  n={args.n}  |  run_id={run_id}")
    print(f"  target branch: {new_branch}")
    print(f"{'='*60}\n")

    # ── Step 1: run tests ──────────────────────────────────────────────────
    print("[1/5] Running self-tests...")
    records = collect_records(args.n)
    if not records:
        print("ERROR: no test records collected. Aborting.")
        sys.exit(1)

    total   = len(records)
    passed  = sum(1 for r in records if r["verdict"] == "pass")
    failed  = total - passed
    avg_score = sum(r["score"] for r in records) / total

    print(f"\n  Results: {passed}/{total} pass  |  {failed} fail  |  avg score {avg_score:.2f}\n")

    # ── Step 2: extract failures ───────────────────────────────────────────
    print("[2/5] Extracting failures...")
    failures = extract_failures(records)
    print(f"  {len(failures)} failures found")

    if not failures:
        print("\nAll tests passed! Nothing to fix.")
        sys.exit(0)

    # ── Step 3: analyse ────────────────────────────────────────────────────
    print("[3/5] Analysing failures with LLM...")
    bugs = analyse_failures(failures)
    bug_report_path = _save_bug_report(bugs, records, run_id)
    print(f"  {len(bugs)} bugs identified → {bug_report_path}")
    print()
    for bug in bugs:
        print(f"  [{bug['id']}] [{bug['severity'].upper()}] {bug['title']}")
        print(f"         files:  {', '.join(bug['affected_files'])}")
        print(f"         cause:  {bug['root_cause']}")
        print(f"         fix:    {bug['fix_description'][:120]}")
        print()

    if not bugs:
        print("LLM found no actionable bugs (all failures may be test-generator errors).")
        sys.exit(0)

    # ── Step 4: auto-fix ───────────────────────────────────────────────────
    print("[4/5] Auto-fixing bugs...")
    all_fixed: dict[str, str] = {}   # rel_path -> latest fixed content

    for bug in bugs:
        if bug["severity"] == "low" and len(bugs) > 5:
            print(f"  Skipping low-severity {bug['id']} (enough high/medium bugs to fix)")
            continue
        print(f"  Fixing {bug['id']}: {bug['title']}...")
        fixed = fix_bug(bug, failures)
        if fixed:
            all_fixed.update(fixed)
            print(f"    Fixed {len(fixed)} file(s): {list(fixed.keys())}")
        else:
            print(f"    Could not fix {bug['id']}")

    if not all_fixed:
        print("No files were fixed. Aborting push.")
        sys.exit(1)

    print(f"\n  Total files to push: {len(all_fixed)}")
    for p in all_fixed:
        print(f"    - {p}")

    # ── Step 5: push ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[5/5] --dry-run mode: skipping push.")
        print("Fixed file paths:", list(all_fixed.keys()))
    else:
        print(f"\n[5/5] Pushing to branch '{new_branch}'...")
        bug_ids = ", ".join(b["id"] for b in bugs if b["severity"] != "low" or len(bugs) <= 5)
        commit_msg = f"fix: self-heal auto-fix [{run_id}] bugs: {bug_ids}"
        ok = _github_push(new_branch, all_fixed, commit_msg)
        if ok:
            print(f"\n✅ Done! Branch: https://github.com/s4s4s4s/Jarvis/tree/{new_branch}")
            print(f"   Commit: {commit_msg}")
        else:
            print("\n❌ Push failed. Check logs above.")
            print("Fixed files are written locally. Run manually:")
            print(f"  git checkout -b {new_branch}")
            print(f"  git add -A")
            print(f"  git commit -m '{commit_msg}'")
            print(f"  git push origin {new_branch}")

    print(f"\n{'='*60}")
    print(f"  Bug report: {bug_report_path}")
    print(f"  Run ID:     {run_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
