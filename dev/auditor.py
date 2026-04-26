"""
dev/auditor.py
Verifier-agent: аудит файлов → list[Finding] → Level 1 верификация → дедупликация.
Usage: python -m dev.auditor brain/ask.py brain/agents/chat.py ...
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from rapidfuzz import fuzz

# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    file: str
    line: int
    type: str           # HARDCODED_STRING | RACE_CONDITION | UNHANDLED_EXCEPTION | RESOURCE_LEAK
    description: str
    suggestion: str
    confidence: float
    status: str = "needs_review"   # confirmed | rejected | needs_review
    reject_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────────

LLAMA_URL   = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_ALIAS = "qwen3-coder"
MAX_TOKENS  = 2048
TEMPERATURE = 0.2
DEDUP_THRESHOLD = 80

SYSTEM_PROMPT = """\
You are a code auditor. Analyze the provided Python source files and find real bugs only.
For each finding output a JSON object on a single line with EXACTLY these keys:
  file, line, type, description, suggestion, confidence
Types allowed: HARDCODED_STRING, RACE_CONDITION, UNHANDLED_EXCEPTION, RESOURCE_LEAK
confidence: float 0.0-1.0
Do NOT output any text outside the JSON lines. Do NOT wrap in markdown fences.
If no issues found, output nothing."""

USER_TEMPLATE = """\
Audit the following files for bugs. Output one JSON object per line, no other text.

{file_blocks}"""


# ──────────────────────────────────────────────────────────────────────────────
# AuditorAgent
# ──────────────────────────────────────────────────────────────────────────────

class AuditorAgent:
    def __init__(
        self,
        llama_url: str = LLAMA_URL,
        model: str = MODEL_ALIAS,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        dedup_threshold: int = DEDUP_THRESHOLD,
    ):
        self.llama_url = llama_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dedup_threshold = dedup_threshold

    # ── public ────────────────────────────────────────────────────────────────

    def audit(self, file_paths: list[str]) -> list[Finding]:
        """Полный цикл: LLM → парсинг → Level 1 верификация → дедуп."""
        prompt = self._build_prompt(file_paths)
        raw_text = self._call_llm(prompt)
        raw_findings = self._parse_findings(raw_text)
        verified = self._verify_facts(raw_findings)
        deduped = self._dedupe(verified)
        return deduped

    # ── private ───────────────────────────────────────────────────────────────

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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        try:
            resp = requests.post(self.llama_url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Не могу подключиться к llama-server. "
                "Запусти C:\\llama\\start-coder.ps1 и повтори."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("llama-server не ответил за 120 секунд — модель ещё грузится?")
        except Exception as e:
            raise RuntimeError(f"Ошибка запроса к LLM: {e}") from e

    def _parse_findings(self, raw: str) -> list[Finding]:
        """Парсим ответ: каждая строка — JSON-объект Finding."""
        findings = []
        required_keys = {"file", "line", "type", "description", "suggestion", "confidence"}

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            if line.startswith("```"):
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
                f = Finding(
                    file=str(data["file"]),
                    line=int(data["line"]),
                    type=str(data.get("type", "UNKNOWN")).upper(),
                    description=str(data["description"]),
                    suggestion=str(data["suggestion"]),
                    confidence=float(data.get("confidence", 0.5)),
                )
                findings.append(f)
            except (ValueError, TypeError):
                continue

        return findings

    def _verify_facts(self, findings: list[Finding]) -> list[Finding]:
        """Level 1 детерминированная верификация."""
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
                f.reject_reason = (
                    f"LINE_OUT_OF_RANGE: line {f.line}, file has {total_lines} lines"
                )
                result.append(f)
                continue

            actual_line = lines[f.line - 1].strip()
            if not actual_line or actual_line.startswith("#"):
                f.status = "needs_review"
                f.reject_reason = "LINE_IS_COMMENT_OR_EMPTY — проверь вручную"
                result.append(f)
                continue

            f.status = "confirmed"
            result.append(f)

        return result

    def _dedupe(self, findings: list[Finding]) -> list[Finding]:
        """
        Дедупликация через rapidfuzz.token_sort_ratio на поле description.
        Оставляет finding с более высоким confidence при дубликатах.
        """
        unique: list[Finding] = []
        for candidate in findings:
            is_dup = False
            for existing in unique:
                score = fuzz.token_sort_ratio(
                    candidate.description, existing.description
                )
                if score >= self.dedup_threshold:
                    if candidate.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(candidate)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        return unique


# ──────────────────────────────────────────────────────────────────────────────
# CLI точка входа
# ──────────────────────────────────────────────────────────────────────────────

def _print_findings(findings: list[Finding]) -> None:
    confirmed    = [f for f in findings if f.status == "confirmed"]
    needs_review = [f for f in findings if f.status == "needs_review"]
    rejected     = [f for f in findings if f.status == "rejected"]

    print(f"\n{'='*60}")
    print(f"  АУДИТ ЗАВЕРШЁН: {len(findings)} raw → "
          f"{len(confirmed)} confirmed, "
          f"{len(needs_review)} needs_review, "
          f"{len(rejected)} rejected")
    print(f"{'='*60}\n")

    for label, group in [
        ("✅ CONFIRMED", confirmed),
        ("⚠️  NEEDS REVIEW", needs_review),
        ("❌ REJECTED", rejected),
    ]:
        if not group:
            continue
        print(f"── {label} ({'─'*(50-len(label))})")
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
    print("Отправляю запрос на llama-server...")

    try:
        findings = agent.audit(files)
    except RuntimeError as e:
        print(f"\n❌ ОШИБКА: {e}")
        sys.exit(1)

    _print_findings(findings)
