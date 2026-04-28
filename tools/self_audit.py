"""tools/self_audit.py

Self-audit tool: Jarvis audits his own source files.

Jarvis runs AuditorAgent on all his own .py files and returns
a structured report. Triggered via tool route: 'auditor.self'.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# All source directories to audit (relative to repo root)
_SOURCE_DIRS = [
    "brain",
    "tools",
    "core",
    "dev",
]

# Files to always skip (generated, migrations, stubs, etc.)
_SKIP_PATTERNS = {
    "__pycache__",
    ".pyc",
    "findings",   # dev/findings — JSON output, not source
}


def _collect_source_files(root: Path) -> list[str]:
    """Collect all .py files from _SOURCE_DIRS, skip junk."""
    files: list[str] = []
    for src_dir in _SOURCE_DIRS:
        src_path = root / src_dir
        if not src_path.exists():
            continue
        for py_file in sorted(src_path.rglob("*.py")):
            rel = str(py_file.relative_to(root))
            if any(skip in rel for skip in _SKIP_PATTERNS):
                continue
            if py_file.stat().st_size < 10:  # skip empty __init__.py
                continue
            files.append(str(py_file))
    return files


def self_audit(
    dirs: list[str] | None = None,
    model: str | None = None,
    confidence_threshold: float = 0.5,
) -> dict:
    """
    Run AuditorAgent on Jarvis own source files.

    Args:
        dirs:                  Override source dirs to audit (e.g. ['brain', 'tools']).
                               Default: all dirs from _SOURCE_DIRS.
        model:                 Override LLM model (default: MODEL_ROUTER from brain.client).
        confidence_threshold:  Only return findings above this confidence (default: 0.5).

    Returns a dict:
        {
          "files_audited":  int,
          "total_findings": int,
          "confirmed":      int,
          "needs_review":   int,
          "rejected":       int,
          "report":         str   <- human-readable summary
          "findings":       list[dict]
        }
    """
    from dev.auditor import AuditorAgent, GENERIC_SYSTEM_PROMPT

    # Detect repo root: go up from this file until we find brain/
    here = Path(__file__).resolve().parent
    root = here
    for _ in range(5):
        if (root / "brain").exists():
            break
        root = root.parent
    else:
        root = Path(os.getcwd())

    # Collect files
    if dirs:
        global _SOURCE_DIRS
        original = _SOURCE_DIRS
        _SOURCE_DIRS = dirs
        files = _collect_source_files(root)
        _SOURCE_DIRS = original
    else:
        files = _collect_source_files(root)

    if not files:
        return {
            "files_audited": 0,
            "total_findings": 0,
            "confirmed": 0,
            "needs_review": 0,
            "rejected": 0,
            "report": "Нет файлов для аудита.",
            "findings": [],
        }

    logger.info("[self_audit] Auditing %d files in %s", len(files), root)

    # Build agent
    kwargs: dict = {"system_prompt": GENERIC_SYSTEM_PROMPT}
    if model:
        kwargs["model"] = model
    agent = AuditorAgent(**kwargs)

    # Run in batches of 5 files to avoid context overflow
    BATCH_SIZE = 5
    all_findings = []
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        logger.info("[self_audit] Batch %d-%d: %s", i+1, i+len(batch), [Path(f).name for f in batch])
        try:
            batch_findings = agent.audit(batch)
            all_findings.extend(batch_findings)
        except Exception as e:
            logger.error("[self_audit] Batch failed: %s", e)

    # Filter by confidence
    all_findings = [f for f in all_findings if f.confidence >= confidence_threshold]

    confirmed    = [f for f in all_findings if f.status == "confirmed"]
    needs_review = [f for f in all_findings if f.status == "needs_review"]
    rejected     = [f for f in all_findings if f.status == "rejected"]

    # Build human-readable report
    lines: list[str] = [
        f"🔍 Self-audit Jarvis: {len(files)} файлов, {len(all_findings)} находок",
        f"✅ Подтверждено: {len(confirmed)}  ⚠️ На проверке: {len(needs_review)}  ❌ Отклонено: {len(rejected)}",
        "",
    ]

    if confirmed:
        lines.append("── ПОДТВЕРЖДЁННЫЕ ПРОБЛЕМЫ:")
        for f in sorted(confirmed, key=lambda x: -x.confidence):
            lines.append(f"  [{f.type}] {f.file}:{f.line}  (conf={f.confidence:.2f})")
            lines.append(f"  Проблема: {f.description}")
            lines.append(f"  Решение:  {f.suggestion}")
            lines.append("")

    if needs_review:
        lines.append("── ТРЕБУЮТ ПРОВЕРКИ:")
        for f in needs_review:
            lines.append(f"  [{f.type}] {f.file}:{f.line}  (conf={f.confidence:.2f})")
            lines.append(f"  {f.description}")
            lines.append("")

    if not confirmed and not needs_review:
        lines.append("✨ Серьёзных проблем не обнаружено.")

    report = "\n".join(lines)

    return {
        "files_audited":  len(files),
        "total_findings": len(all_findings),
        "confirmed":      len(confirmed),
        "needs_review":   len(needs_review),
        "rejected":       len(rejected),
        "report":         report,
        "findings":       [
            {
                "file":        f.file,
                "line":        f.line,
                "type":        f.type,
                "description": f.description,
                "suggestion":  f.suggestion,
                "confidence":  f.confidence,
                "status":      f.status,
            }
            for f in all_findings
        ],
    }
