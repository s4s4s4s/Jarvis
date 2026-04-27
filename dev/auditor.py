"""
dev/auditor.py
Verifier-agent: аудит файлов → list[Finding] → Level 1 верификация → дедупликация.
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

from brain.client import chat, MODEL_ROUTER
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


AUDIT_MODEL     = MODEL_ROUTER
TEMPERATURE     = 0.2
DEDUP_THRESHOLD = 80

# Default prompt — Jarvis-specific rules
DEFAULT_SYSTEM_PROMPT = """\
You are a strict code auditor. Analyze the provided Python source files and report ALL issues.
For each finding output a JSON object on a single line with EXACTLY these keys:
  file, line, type, description, suggestion, confidence

Types allowed:
  HARDCODED_STRING    - user-facing string literals that should be LLM-generated, not hardcoded.
                        This project philosophy: ALL user-facing strings must be LLM-generated.
  RACE_CONDITION      - shared state accessed from multiple threads without synchronization
  UNHANDLED_EXCEPTION - bare except, silent exception swallowing, missing logging on catch
  RESOURCE_LEAK       - file handles, threads, connections not properly closed/joined

IMPORTANT exclusions - do NOT flag these as HARDCODED_STRING:
  - Tool registry mappings (_TOOL_MAP, dispatch dicts) — these are architecture, not user strings
  - Exception fallback strings that are already guarded by logger.error() above them
  - Format string templates used as constants (e.g. _FALLBACK_ERROR = "..{error}")
  - Code comments
  - Internal label strings passed to logging/debugging functions

confidence: float 0.0-1.0
Do NOT output any text outside the JSON lines. Do NOT wrap in markdown fences.
If no issues found, output nothing."""

# Generic prompt — for external/user code, no Jarvis-specific rules
GENERIC_SYSTEM_PROMPT = """\
You are a strict code auditor. Analyze the provided Python source files and report ALL issues.
For each finding output a JSON object on a single line with EXACTLY these keys:
  file, line, type, description, suggestion, confidence

Types allowed:
  BUG                 - logic error, wrong algorithm, off-by-one, incorrect condition
  SECURITY            - injection, hardcoded secrets, insecure defaults, unvalidated input
  UNHANDLED_EXCEPTION - bare except, silent exception swallowing, missing logging on catch
  RESOURCE_LEAK       - file handles, threads, connections, sockets not properly closed/joined
  PERFORMANCE         - unnecessary loops, blocking calls in async context, inefficient data structures
  RACE_CONDITION      - shared state accessed from multiple threads without synchronization

confidence: float 0.0-1.0
Do NOT output any text outside the JSON lines. Do NOT wrap in markdown fences.
If no issues found, output nothing."""

USER_TEMPLATE = """\
Audit the following files. Output one JSON object per line, no other text.

{file_blocks}"""


class AuditorAgent:
    def __init__(
        self,
        model: str = AUDIT_MODEL,
        temperature: float = TEMPERATURE,
        dedup_threshold: int = DEDUP_THRESHOLD,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.dedup_threshold = dedup_threshold
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    def audit(self, file_paths: list[str]) -> list[Finding]:
        prompt = self._build_prompt(file_paths)
        raw_text = self._call_llm(prompt)
        raw_findings = self._parse_findings(raw_text)
        verified = self._verify_facts(raw_findings)
        return self._dedupe(verified)

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

    @staticmethod
    def _iter_objects(raw: str):
        """
        Yield all dicts from LLM output, handling three formats:
          1. One JSON object per line: {...}\n{...}
          2. JSON array on one or multiple lines: [{...}, {...}]
          3. Mixed garbage with embedded JSON objects
        """
        # Try full parse first (handles array or single object spanning multiple lines)
        stripped = raw.strip()
        # Remove markdown fences if present
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner = []
            in_block = False
            for ln in lines:
                if ln.startswith("```") and not in_block:
                    in_block = True
                    continue
                if ln.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(ln)
            stripped = "\n".join(inner).strip()

        # Attempt full-document parse (array or single object)
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        yield item
                return
            if isinstance(parsed, dict):
                yield parsed
                return
        except json.JSONDecodeError:
            pass

        # Fall back to line-by-line parsing
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
                        if isinstance(item, dict):
                            yield item
                continue
            except json.JSONDecodeError:
                pass
            # Last resort: extract first {...} from line
            match = re.search(r"\{[^{}]+\}", line)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, dict):
                        yield data
                except json.JSONDecodeError:
                    pass

    def _parse_findings(self, raw: str) -> list[Finding]:
        findings = []
        required_keys = {"file", "line", "type", "description", "suggestion", "confidence"}
        for data in self._iter_objects(raw):
            if required_keys - set(data.keys()):
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

    def _verify_facts(self, findings: list[Finding]) -> list[Finding]:
        result = []
        for f in findings:
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
                f.status = "needs_review"
                f.reject_reason = "LINE_IS_COMMENT_OR_EMPTY"
                result.append(f)
                continue
            f.status = "confirmed"
            result.append(f)
        return result

    def _dedupe(self, findings: list[Finding]) -> list[Finding]:
        unique: list[Finding] = []
        for candidate in findings:
            is_dup = False
            for existing in unique:
                if fuzz.token_sort_ratio(candidate.description, existing.description) >= self.dedup_threshold:
                    if candidate.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(candidate)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        return unique


def _print_findings(findings: list[Finding]) -> None:
    confirmed    = [f for f in findings if f.status == "confirmed"]
    needs_review = [f for f in findings if f.status == "needs_review"]
    rejected     = [f for f in findings if f.status == "rejected"]
    print(f"\n{'='*60}")
    print(f"  АУДИТ: {len(findings)} raw → {len(confirmed)} confirmed, "
          f"{len(needs_review)} needs_review, {len(rejected)} rejected")
    print(f"{'='*60}\n")
    for label, group in [("✅ CONFIRMED", confirmed), ("⚠️ NEEDS REVIEW", needs_review), ("❌ REJECTED", rejected)]:
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m dev.auditor <file1> [file2 ...]")
        sys.exit(1)
    files = sys.argv[1:]
    agent = AuditorAgent()
    print(f"Аудитирую {len(files)} файл(ов): {', '.join(files)}")
    print(f"Модель: {AUDIT_MODEL} (через Ollama)")
    try:
        findings = agent.audit(files)
    except RuntimeError as e:
        print(f"\n❌ ОШИБКА: {e}")
        sys.exit(1)
    _print_findings(findings)
