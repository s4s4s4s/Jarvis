"""
dev/auditor.py
Verifier-agent: аудит файлов → list[Finding]
  → L1 (fact-check: file/line/comment)
  → confidence threshold filter
  → L2 (semantic re-verify via MODEL_HEAVY)
  → дедупликация
Usage: python -m dev.auditor brain/ask.py brain/agents/chat.py ...

Backend: Ollama (brain.client) — тот же что и основной Jarvis, нет конфликта по VRAM.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from brain.client import chat, MODEL_ROUTER, MODEL_HEAVY
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    file: str
    line: int
    type: str
    description: str
    suggestion: str
    confidence: float
    status: str = "needs_review"
    reject_reason: Optional[str] = None


# L1 (scan) uses fast model — wide net
AUDIT_MODEL      = MODEL_ROUTER
# L2 (verify) uses heavy model — precision filter
VERIFY_MODEL     = MODEL_HEAVY

TEMPERATURE      = 0.2
DEDUP_THRESHOLD  = 80

# Findings below this threshold go straight to needs_review without L2
# Findings at or above pass to L2 for semantic verification
CONFIDENCE_MIN_L2    = 0.65   # minimum to enter L2 (below → needs_review)
CONFIDENCE_MIN_SHOW  = 0.50   # minimum to show at all (below → rejected)

# ---------------------------------------------------------------------------
# Few-shot examples embedded in prompts to teach the model what NOT to flag
# ---------------------------------------------------------------------------
_FEW_SHOT_JARVIS = """
## EXAMPLES OF BAD FINDINGS (do NOT produce these):

  BAD: {"type": "HARDCODED_STRING", "description": "WMO weather condition codes are hardcoded",
        "confidence": 0.8}
  WHY WRONG: WMO codes are a lookup table / data constant, not a user-facing string.

  BAD: {"type": "HARDCODED_STRING", "description": "Russian text in timer label",
        "confidence": 0.7}
  WHY WRONG: Language of internal labels is not a bug. Project language is Russian.

  BAD: {"type": "SECURITY", "description": "JARVIS_ROOT path is hardcoded",
        "confidence": 0.75}
  WHY WRONG: JARVIS_ROOT is a config constant read from env, not a secret.

  BAD: {"type": "UNHANDLED_EXCEPTION", "description": "Exception swallowed at line 42",
        "confidence": 0.6}
  WHY WRONG: Line 42 has logger.error() above the except — it IS logged.

## EXAMPLES OF GOOD FINDINGS (DO produce these):

  GOOD: {"type": "UNHANDLED_EXCEPTION", "description": "bare except: pass swallows all errors silently",
         "confidence": 0.95}
  WHY CORRECT: No logging, no re-raise, error is lost completely.

  GOOD: {"type": "RACE_CONDITION", "description": "_cache mutated outside _lock in add_fact()",
         "confidence": 0.9}
  WHY CORRECT: _cache is shared between threads and the mutation happens before _save() under no lock.

  GOOD: {"type": "RESOURCE_LEAK", "description": "file opened with open() without with-statement",
         "confidence": 0.85}
  WHY CORRECT: If exception occurs between open() and close(), file handle leaks.
"""

_FEW_SHOT_GENERIC = """
## EXAMPLES OF BAD FINDINGS (do NOT produce these):

  BAD: {"type": "SECURITY", "description": "WMO codes or country codes hardcoded",
        "confidence": 0.7}
  WHY WRONG: Lookup tables and data constants are not secrets.

  BAD: {"type": "SECURITY", "description": "String in another language is hardcoded",
        "confidence": 0.6}
  WHY WRONG: i18n text is not a security issue.

  BAD: {"type": "UNHANDLED_EXCEPTION", "description": "Exception caught at line X",
        "confidence": 0.5}
  WHY WRONG: If the except block has logging, it is handled. Check before reporting.

## EXAMPLES OF GOOD FINDINGS (DO produce these):

  GOOD: {"type": "SECURITY", "description": "API key hardcoded as string literal: api_key = 'sk-...'",
         "confidence": 0.99}
  WHY CORRECT: Actual secret credential in source code.

  GOOD: {"type": "BUG", "description": "off-by-one: range(len(x)) should be range(len(x)-1)",
         "confidence": 0.9}
  WHY CORRECT: Logic error clearly visible in code.

  GOOD: {"type": "RESOURCE_LEAK", "description": "socket not closed in error path",
         "confidence": 0.85}
  WHY CORRECT: try/except exists but finally block is missing.
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
"""\
You are a strict code auditor. Analyze the provided Python source files and report ALL issues.
For each finding output a JSON object on a single line with EXACTLY these keys:
  file, line, type, description, suggestion, confidence

Types allowed:
  HARDCODED_STRING    - user-facing string literals that should be LLM-generated, not hardcoded.
                        This project philosophy: ALL user-facing strings must be LLM-generated.
  RACE_CONDITION      - shared state accessed from multiple threads without synchronization
  UNHANDLED_EXCEPTION - bare except, silent exception swallowing, missing logging on catch
  RESOURCE_LEAK       - file handles, threads, connections not properly closed/joined

IMPORTANT exclusions — do NOT flag these as HARDCODED_STRING:
  - Tool registry mappings (_TOOL_MAP, dispatch dicts) — these are architecture, not user strings
  - Exception fallback strings already guarded by logger.error() above them
  - Format string templates used as constants
  - Code comments
  - Internal label strings passed to logging/debugging functions
  - Lookup tables, data constants, WMO codes, mapping dicts
  - Russian text — the project language is Russian, this is not a bug

confidence: float 0.0–1.0. Be conservative: only report confidence ≥ 0.65 for real issues.
Do NOT output any text outside the JSON lines. Do NOT wrap in markdown fences.
If no issues found, output nothing.
"""
+ _FEW_SHOT_JARVIS
)

GENERIC_SYSTEM_PROMPT = (
"""\
You are a strict code auditor. Analyze the provided Python source files and report ALL issues.
For each finding output a JSON object on a single line with EXACTLY these keys:
  file, line, type, description, suggestion, confidence

Types allowed:
  BUG                 - logic error, wrong algorithm, off-by-one, incorrect condition
  SECURITY            - injection, hardcoded secrets (API keys, passwords, tokens), insecure defaults
  UNHANDLED_EXCEPTION - bare except, silent exception swallowing, missing logging on catch
  RESOURCE_LEAK       - file handles, threads, connections, sockets not properly closed/joined
  PERFORMANCE         - unnecessary loops, blocking calls in async context, inefficient data structures
  RACE_CONDITION      - shared state accessed from multiple threads without synchronization

IMPORTANT: SECURITY findings must involve actual secrets (passwords, API keys, tokens), NOT:
  - Hardcoded lookup tables (WMO codes, language codes, country codes)
  - i18n / localization text in any language
  - Data constants, mapping dicts
  - Internal log messages

confidence: float 0.0–1.0. Be conservative: only report confidence ≥ 0.65 for real issues.
Do NOT output any text outside the JSON lines. Do NOT wrap in markdown fences.
If no issues found, output nothing.
"""
+ _FEW_SHOT_GENERIC
)

USER_TEMPLATE = """\
Audit the following files. Output one JSON object per line, no other text.

{file_blocks}"""

# ---------------------------------------------------------------------------
# Level-2 semantic re-verification prompt (MODEL_HEAVY)
# ---------------------------------------------------------------------------
_VERIFY_SYSTEM = """\
You are a senior code reviewer doing a SECOND-PASS verification of audit findings.
For each finding you will be shown:
  1. The finding (type, description, suggestion)
  2. The ACTUAL code context ±5 lines around the reported line

Your job: decide if this finding is VALID or INVALID.

A finding is INVALID if ANY of the following are true:
- The described problem does NOT exist in the shown code
- The code context clearly shows the issue is already handled (logging, with-statement, try/finally)
- The “issue” is a data constant, lookup table, mapping dict, or i18n / localization text
- The “security issue” is NOT an actual secret (API key / password / token / private key)
- The description contradicts what the code actually does
- The finding describes a non-problem (e.g. “Russian text”, “configuration constant”)

A finding is VALID only if:
- The exact problem is clearly present in the shown code
- It is a genuine: unhandled exception, resource leak, race condition, real security secret, or logic bug

Examples:
  INVALID: type=SECURITY, desc="WMO codes hardcoded" → those are data constants, not secrets
  INVALID: type=UNHANDLED_EXCEPTION, desc="caught at line 38" → context shows logger.error() is present
  VALID:   type=UNHANDLED_EXCEPTION, desc="bare except: pass" → context shows `except: pass` with no logging
  VALID:   type=RESOURCE_LEAK, desc="file opened without with" → context shows open() not in with-block

Output ONLY a JSON array, one object per finding:
  [{"index": 0, "verdict": "VALID", "reason": "<one sentence>"},
   {"index": 1, "verdict": "INVALID", "reason": "<one sentence>"}]
No other text outside the JSON array."""


class AuditorAgent:
    def __init__(
        self,
        model: str = AUDIT_MODEL,
        verify_model: str = VERIFY_MODEL,
        temperature: float = TEMPERATURE,
        dedup_threshold: int = DEDUP_THRESHOLD,
        system_prompt: str | None = None,
        confidence_min_l2: float = CONFIDENCE_MIN_L2,
        confidence_min_show: float = CONFIDENCE_MIN_SHOW,
    ):
        self.model             = model
        self.verify_model      = verify_model
        self.temperature       = temperature
        self.dedup_threshold   = dedup_threshold
        self.system_prompt     = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.confidence_min_l2   = confidence_min_l2
        self.confidence_min_show = confidence_min_show

    def audit(self, file_paths: list[str]) -> list[Finding]:
        prompt       = self._build_prompt(file_paths)
        raw_text     = self._call_llm(prompt)
        raw_findings = self._parse_findings(raw_text)
        after_conf   = self._filter_confidence(raw_findings)   # confidence threshold
        after_l1     = self._verify_facts(after_conf)           # L1: file/line/comment
        after_l2     = self._verify_semantic(after_l1)          # L2: MODEL_HEAVY cross-check
        return self._dedupe(after_l2)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, file_paths: list[str]) -> str:
        blocks = []
        for fp in file_paths:
            path = Path(fp)
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8")
            numbered = "\n".join(
                f"{i+1:4d} | {line}"
                for i, line in enumerate(source.splitlines())
            )
            blocks.append(f"### FILE: {fp}\n```python\n{numbered}\n```")
        return USER_TEMPLATE.format(file_blocks="\n\n".join(blocks))

    def _call_llm(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]
        result = chat(
            self.model,
            messages,
            options={"temperature": self.temperature, "num_ctx": 16384},
        )
        if not result:
            raise RuntimeError(
                "Ollama вернула пустой ответ. "
                "Убедись что Ollama запущена: ollama list"
            )
        return result

    # ------------------------------------------------------------------
    # Confidence threshold  (pre-L1 filter)
    # ------------------------------------------------------------------

    def _filter_confidence(self, findings: list[Finding]) -> list[Finding]:
        """
        Split findings by confidence before any verification:
          < confidence_min_show  → rejected immediately
          confidence_min_show..confidence_min_l2  → needs_review (skip L2, shown as warnings)
          ≥ confidence_min_l2  → pass to L1 + L2 pipeline
        """
        result = []
        for f in findings:
            if f.confidence < self.confidence_min_show:
                f.status = "rejected"
                f.reject_reason = (
                    f"CONFIDENCE_TOO_LOW: {f.confidence:.2f} < {self.confidence_min_show}"
                )
                logger.debug(
                    "[Auditor] Rejected (low conf %.2f): [%s] %s:%d",
                    f.confidence, f.type, f.file, f.line,
                )
            elif f.confidence < self.confidence_min_l2:
                f.status = "needs_review"
                f.reject_reason = (
                    f"CONFIDENCE_BELOW_L2: {f.confidence:.2f} — manual check recommended"
                )
                logger.debug(
                    "[Auditor] needs_review (conf %.2f): [%s] %s:%d",
                    f.confidence, f.type, f.file, f.line,
                )
            # else: keep default status, will go through L1/L2
            result.append(f)
        return result

    # ------------------------------------------------------------------
    # L1: factual verification (file / line / comment)
    # ------------------------------------------------------------------

    def _verify_facts(self, findings: list[Finding]) -> list[Finding]:
        """Level-1: only process findings still in needs_review/default state."""
        result = []
        for f in findings:
            # Already decided by confidence filter — don’t re-process
            if f.status in ("rejected", "needs_review"):
                result.append(f)
                continue
            path = Path(f.file)
            if not path.exists():
                f.status = "rejected"
                f.reject_reason = f"FILE_NOT_FOUND: {f.file}"
                result.append(f)
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            total_lines = len(lines)
            if f.line < 1 or f.line > total_lines:
                f.status = "rejected"
                f.reject_reason = f"LINE_OUT_OF_RANGE: line {f.line}, file has {total_lines} lines"
                result.append(f)
                continue
            actual_line = lines[f.line - 1].strip()
            if not actual_line or actual_line.startswith("#"):
                f.status = "rejected"
                f.reject_reason = "LINE_IS_COMMENT_OR_EMPTY"
                result.append(f)
                continue
            f.status = "confirmed"  # tentative — re-checked in L2
            result.append(f)
        return result

    # ------------------------------------------------------------------
    # L2: semantic re-verification (MODEL_HEAVY)
    # ------------------------------------------------------------------

    def _verify_semantic(self, findings: list[Finding]) -> list[Finding]:
        """
        Level-2: batch all L1-confirmed findings to MODEL_HEAVY.
        Shows 10-line context around each reported line.
        """
        candidates = [f for f in findings if f.status == "confirmed"]
        rest       = [f for f in findings if f.status != "confirmed"]

        if not candidates:
            return findings

        items = []
        for idx, f in enumerate(candidates):
            path = Path(f.file)
            lines = path.read_text(encoding="utf-8").splitlines()
            lo = max(0, f.line - 5)          # 5 lines before
            hi = min(len(lines), f.line + 5) # 5 lines after
            context_lines = "\n".join(
                f"{lo+i+1:4d} | {lines[lo+i]}" for i in range(hi - lo)
            )
            items.append({
                "index": idx,
                "file": f.file,
                "type": f.type,
                "description": f.description,
                "suggestion": f.suggestion,
                "confidence": f.confidence,
                "reported_line": f.line,
                "code_context": context_lines,
            })

        user_content = json.dumps(items, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user",   "content": user_content},
        ]

        try:
            logger.info(
                "[Auditor] L2 verify: %d candidates → %s",
                len(candidates), self.verify_model,
            )
            raw = chat(
                self.verify_model,   # ← MODEL_HEAVY here
                messages,
                options={"temperature": 0.0, "num_ctx": 8192},
            )
            verdicts = self._parse_verdicts(raw)
        except Exception as e:
            logger.warning(
                "[Auditor] L2 verify failed (%s) — skipping L2, keeping L1 results", e
            )
            return findings

        for v in verdicts:
            idx = v.get("index")
            if idx is None or not isinstance(idx, int) or idx >= len(candidates):
                continue
            f = candidates[idx]
            verdict = str(v.get("verdict", "")).upper()
            reason  = str(v.get("reason", ""))
            if verdict == "INVALID":
                f.status = "rejected"
                f.reject_reason = f"L2_SEMANTIC: {reason}"
                logger.info(
                    "[Auditor] L2 rejected [%s] %s:%d — %s",
                    f.type, f.file, f.line, reason,
                )
            else:
                logger.info(
                    "[Auditor] L2 confirmed ✅ [%s] %s:%d",
                    f.type, f.file, f.line,
                )

        return candidates + rest

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_objects(raw: str):
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner, in_block = [], False
            for ln in lines:
                if ln.startswith("```") and not in_block:
                    in_block = True; continue
                if ln.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(ln)
            stripped = "\n".join(inner).strip()

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict): yield item
                return
            if isinstance(parsed, dict):
                yield parsed; return
        except json.JSONDecodeError:
            pass

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#") or line.startswith("```"):
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict): yield item
                continue
            except json.JSONDecodeError:
                pass
            m = re.search(r"\{[^{}]+\}", line)
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, dict): yield data
                except json.JSONDecodeError:
                    pass

    def _parse_findings(self, raw: str) -> list[Finding]:
        findings = []
        required = {"file", "line", "type", "description", "suggestion", "confidence"}
        for data in self._iter_objects(raw):
            if required - set(data.keys()):
                logger.debug("[Auditor] Skipping object missing keys: %s", data)
                continue
            try:
                findings.append(Finding(
                    file=str(data["file"]),
                    line=int(data["line"]),
                    type=str(data.get("type", "UNKNOWN")).upper(),
                    description=str(data["description"]),
                    suggestion=str(data["suggestion"]),
                    confidence=float(data.get("confidence", 0.5)),
                ))
            except (ValueError, TypeError) as e:
                logger.debug("[Auditor] Skipping malformed finding: %s — %s", data, e)
        return findings

    @staticmethod
    def _parse_verdicts(raw: str) -> list[dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            inner, in_block = [], False
            for ln in lines:
                if ln.startswith("```") and not in_block:
                    in_block = True; continue
                if ln.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(ln)
            raw = "\n".join(inner).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedupe(self, findings: list[Finding]) -> list[Finding]:
        unique: list[Finding] = []
        for candidate in findings:
            is_dup = False
            for existing in unique:
                if fuzz.token_sort_ratio(
                    candidate.description, existing.description
                ) >= self.dedup_threshold:
                    if candidate.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(candidate)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        return unique


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def _print_findings(findings: list[Finding]) -> None:
    confirmed    = [f for f in findings if f.status == "confirmed"]
    needs_review = [f for f in findings if f.status == "needs_review"]
    rejected     = [f for f in findings if f.status == "rejected"]
    print(f"\n{'='*60}")
    print(
        f"  АУДИТ: {len(findings)} raw → "
        f"{len(confirmed)} confirmed, "
        f"{len(needs_review)} needs_review, "
        f"{len(rejected)} rejected"
    )
    print(f"{'='*60}\n")
    for label, group in [
        ("✅ CONFIRMED",    confirmed),
        ("⚠️  NEEDS REVIEW", needs_review),
        ("❌ REJECTED",     rejected),
    ]:
        if not group:
            continue
        print(f"── {label}")
        for f in group:
            print(f"  [{f.type}] {f.file}:{f.line}  conf={f.confidence:.2f}")
            print(f"  DESC: {f.description}")
            print(f"  FIX:  {f.suggestion}")
            if f.reject_reason:
                print(f"  WHY:  {f.reject_reason}")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m dev.auditor <file1> [file2 ...]")
        sys.exit(1)
    files = sys.argv[1:]
    agent = AuditorAgent()
    print(f"Аудитирую {len(files)} файл(ов): {', '.join(files)}")
    print(f"L1-модель (scan):   {AUDIT_MODEL}")
    print(f"L2-модель (verify): {VERIFY_MODEL}")
    print(f"Confidence порог: show={CONFIDENCE_MIN_SHOW}, L2={CONFIDENCE_MIN_L2}")
    try:
        findings = agent.audit(files)
    except RuntimeError as e:
        print(f"\n❌ ОШИБКА: {e}")
        sys.exit(1)
    _print_findings(findings)
