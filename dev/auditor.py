"""
dev/auditor.py
Verifier-agent: аудит файлов → list[Finding] → Level 1 верификация → дедупликация.
Usage: python -m dev.auditor brain/ask.py brain/agents/chat.py ...

Backend: Ollama (brain.client) — тот же что и основной Jarvis, нет конфликта по VRAM.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from brain.client import chat, MODEL_ROUTER
from rapidfuzz import fuzz


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

SYSTEM_PROMPT = """\
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

USER_TEMPLATE = """\
Audit the following files. Output one JSON object per line, no other text.
Pay special attention to:
- Hardcoded user-facing strings in responses, fallbacks, error messages (HARDCODED_STRING)
- Silent exception handlers with no logging (UNHANDLED_EXCEPTION)
- Race conditions on shared state (RACE_CONDITION)
- Unclosed resources (RESOURCE_LEAK)

{file_blocks}"""


class AuditorAgent:
    def __init__(
        self,
        model: str = AUDIT_MODEL,
        temperature: float = TEMPERATURE,
        dedup_threshold: int = DEDUP_THRESHOLD,
    ):
        self.model = model
        self.temperature = temperature
        self.dedup_threshold = dedup_threshold

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
            {"role": "system", "content": SYSTEM_PROMPT},
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

    def _parse_findings(self, raw: str) -> list[Finding]:
        findings = []
        required_keys = {"file", "line", "type", "description", "suggestion", "confidence"}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#") or line.startswith("```"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", line)
                if not match:
                    continue
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    continue
            if required_keys - set(data.keys()):
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
            except (ValueError, TypeError):
                continue
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
