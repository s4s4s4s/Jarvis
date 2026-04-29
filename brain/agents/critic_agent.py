"""brain/agents/critic_agent.py

Independent CriticAgent — reviews code written by the code agent.

Key principle: the AUTHOR and the CRITIC are separate LLM calls with
different system prompts. The critic does not know who wrote the code.
It only knows what the code is SUPPOSED to do and finds every flaw.

Returns a CriticResult with:
  - passed: bool             — True if code is shippable
  - issues: list[Issue]      — every confirmed problem
  - fixed_code: str | None   — fixed version if critic also fixed it

Used by executor._run_code_with_critic() loop.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from brain.client import chat, MODEL_HEAVY
from brain.tools.sandbox import sandbox_audit

logger = logging.getLogger(__name__)

# How many critic→fix iterations before we give up and ship best version
MAX_CRITIC_ROUNDS = 6

_CRITIC_SYSTEM = """\
You are a senior software engineer doing a ruthless code review.
You did NOT write this code. Your job is to find every flaw.

Review the code for ALL of the following:
  1. Runtime errors / crashes / unhandled exceptions
  2. Logic bugs — wrong algorithm, off-by-one, incorrect conditions
  3. Missing edge cases (empty input, None, zero, large values)
  4. Security issues (hardcoded secrets, injection, unsafe eval/exec)
  5. Missing imports or undefined names
  6. Broken dependencies between functions/classes
  7. The code does NOT do what the task description says
  8. Poor error handling that will cause silent failures in production

For each issue found, output one line:
  [SEVERITY] line N: description — fix: concrete fix instruction

SEVERITY must be one of: CRITICAL / HIGH / MEDIUM / LOW

If there are NO issues, output exactly one line:
  LGTM

Do NOT output anything else. No prose. No markdown. Just issue lines or LGTM.
"""

_FIXER_SYSTEM = """\
You are an expert Python developer fixing code based on a review.
You receive:
  1. The task description (what the code should do)
  2. The current code
  3. A list of issues from a code reviewer

Fix ALL listed issues. Return ONLY the complete fixed Python code
inside a ```python ... ``` block. No explanation. No prose.
Do not remove any working functionality.
Do not introduce new features not mentioned in the task.
"""


@dataclass
class Issue:
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW
    line: int
    description: str
    fix: str

    @property
    def is_blocking(self) -> bool:
        return self.severity in ("CRITICAL", "HIGH")


@dataclass
class CriticResult:
    passed: bool
    issues: list[Issue] = field(default_factory=list)
    fixed_code: str | None = None
    rounds: int = 0
    sandbox_report: str = ""

    def blocking_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.is_blocking]

    def summary(self) -> str:
        if self.passed:
            return f"✅ Critic passed after {self.rounds} round(s)."
        blocking = len(self.blocking_issues())
        total = len(self.issues)
        return (
            f"❌ Critic: {blocking} blocking / {total} total issues "
            f"after {self.rounds} round(s)."
        )


def _parse_issues(raw: str) -> list[Issue]:
    """Parse critic output into Issue objects."""
    issues: list[Issue] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "LGTM":
            continue
        # [SEVERITY] line N: description — fix: ...
        m = re.match(
            r"\[(CRITICAL|HIGH|MEDIUM|LOW)\]\s+line\s+(\d+):\s+(.+?)(?:\s+[—\-–]+\s+fix:\s+(.+))?",
            line, re.IGNORECASE,
        )
        if m:
            issues.append(Issue(
                severity=m.group(1).upper(),
                line=int(m.group(2)),
                description=m.group(3).strip(),
                fix=m.group(4).strip() if m.group(4) else "see description",
            ))
        else:
            # loose format — treat whole line as HIGH issue
            if len(line) > 10:
                issues.append(Issue(
                    severity="HIGH",
                    line=0,
                    description=line[:200],
                    fix="see description",
                ))
    return issues


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner, in_block = [], False
        for ln in lines:
            if ln.startswith("```") and not in_block:
                in_block = True
                continue
            if ln.startswith("```") and in_block:
                break
            if in_block:
                inner.append(ln)
        return "\n".join(inner)
    return text


def _critic_review(code: str, task_goal: str) -> list[Issue]:
    """Ask the critic to review code. Returns list of issues."""
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": (
            f"Task description: {task_goal}\n\n"
            f"Code to review:\n```python\n{code}\n```"
        )},
    ]
    raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.05, "num_ctx": 16384})
    return _parse_issues(raw)


def _fix_code(code: str, task_goal: str, issues: list[Issue]) -> str:
    """Ask fixer LLM to fix all issues. Returns fixed code."""
    issue_list = "\n".join(
        f"  [{i.severity}] line {i.line}: {i.description} — fix: {i.fix}"
        for i in issues
    )
    messages = [
        {"role": "system", "content": _FIXER_SYSTEM},
        {"role": "user", "content": (
            f"Task: {task_goal}\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Issues to fix:\n{issue_list}"
        )},
    ]
    raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.05, "num_ctx": 32768})
    fixed = _strip_fences(raw)
    return fixed if len(fixed.strip()) > 50 else code  # safety: never return empty


def review_and_fix(code: str, task_goal: str, max_rounds: int = MAX_CRITIC_ROUNDS) -> CriticResult:
    """
    Main entry point.
    Runs critic → fix loop until:
      - no blocking issues (CRITICAL/HIGH) found, OR
      - sandbox passes, OR
      - max_rounds reached (returns best version found so far)

    Returns CriticResult with passed=True if code is shippable.
    """
    current_code = code
    best_code = code
    best_issue_count = 999
    last_issues: list[Issue] = []

    for round_num in range(1, max_rounds + 1):
        logger.info("[Critic] Round %d/%d for task: %s", round_num, max_rounds, task_goal[:60])

        # Step 1: sandbox runtime check
        sandbox_report, sandbox_has_issues = sandbox_audit(current_code)
        logger.info("[Critic][Sandbox] %s", sandbox_report[:100])

        if sandbox_has_issues:
            # Sandbox found runtime crash — inject as CRITICAL issue
            crash_issue = Issue(
                severity="CRITICAL",
                line=0,
                description=f"Runtime crash: {sandbox_report[:200]}",
                fix="fix the runtime error shown above",
            )
            issues = [crash_issue] + _critic_review(current_code, task_goal)
        else:
            # Step 2: static LLM review
            issues = _critic_review(current_code, task_goal)

        last_issues = issues
        blocking = [i for i in issues if i.is_blocking]

        logger.info(
            "[Critic] Round %d: %d total issues, %d blocking",
            round_num, len(issues), len(blocking),
        )

        # Track best version (fewest issues)
        if len(issues) < best_issue_count:
            best_issue_count = len(issues)
            best_code = current_code

        # Passed: no blocking issues
        if not blocking:
            logger.info("[Critic] ✅ No blocking issues — passed at round %d", round_num)
            return CriticResult(
                passed=True,
                issues=issues,
                fixed_code=current_code,
                rounds=round_num,
                sandbox_report=sandbox_report,
            )

        # Not passed but out of rounds
        if round_num == max_rounds:
            logger.warning("[Critic] ❌ Max rounds reached with %d blocking issues", len(blocking))
            break

        # Fix and continue
        logger.info("[Critic] Fixing %d blocking issues...", len(blocking))
        fixed = _fix_code(current_code, task_goal, blocking)
        if fixed == current_code:
            logger.warning("[Critic] Fixer returned identical code — stopping early")
            break
        current_code = fixed

    # Return best version even if not fully passing
    return CriticResult(
        passed=False,
        issues=last_issues,
        fixed_code=best_code,
        rounds=max_rounds,
        sandbox_report=sandbox_report if 'sandbox_report' in dir() else "",
    )
