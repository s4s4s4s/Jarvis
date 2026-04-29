"""brain/agents/project_context.py

Persistent project state across conversations.
Stores: files written, open tasks, test results, architecture decisions.

Usage:
    ctx = ProjectContext.load()          # load or create fresh
    ctx.add_file("output/app.py", code)  # register a written file
    ctx.open_task("Add auth module")     # track what's left to do
    ctx.close_task("Add auth module")    # mark done
    ctx.record_test("test_app", True)    # record test result
    ctx.save()                           # persist to disk
    summary = ctx.summary()             # inject into planner prompt
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_PATH = Path("project_state.json")


class ProjectContext:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        d = data or {}
        self.project_name: str = d.get("project_name", "")
        self.description: str = d.get("description", "")
        # {rel_path: {"written_at": iso, "size": int, "summary": str}}
        self.files: dict[str, dict] = d.get("files", {})
        # [{"task": str, "status": open|done|failed, "created_at": iso}]
        self.tasks: list[dict] = d.get("tasks", [])
        # [{"name": str, "passed": bool, "at": iso, "details": str}]
        self.test_results: list[dict] = d.get("test_results", [])
        # [str]  — architecture notes, key decisions
        self.decisions: list[str] = d.get("decisions", [])
        # last N errors seen during runs
        self.recent_errors: list[str] = d.get("recent_errors", [])
        self._max_errors = 20

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path = _STATE_PATH) -> "ProjectContext":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("[ProjectContext] Loaded from %s", path)
                return cls(data)
            except Exception as e:
                logger.warning("[ProjectContext] Failed to load state: %s — starting fresh", e)
        return cls()

    def save(self, path: Path = _STATE_PATH) -> None:
        try:
            path.write_text(
                json.dumps(self._to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("[ProjectContext] Saved to %s", path)
        except Exception as e:
            logger.error("[ProjectContext] Failed to save state: %s", e)

    def _to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "description": self.description,
            "files": self.files,
            "tasks": self.tasks,
            "test_results": self.test_results,
            "decisions": self.decisions,
            "recent_errors": self.recent_errors,
        }

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def add_file(self, rel_path: str, code: str, summary: str = "") -> None:
        self.files[rel_path] = {
            "written_at": _now(),
            "size": len(code),
            "summary": summary or _one_line(code),
        }
        logger.debug("[ProjectContext] Registered file: %s", rel_path)

    def remove_file(self, rel_path: str) -> None:
        self.files.pop(rel_path, None)

    def file_list(self) -> list[str]:
        return list(self.files.keys())

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def open_task(self, description: str) -> None:
        # avoid duplicates
        if not any(t["task"] == description and t["status"] == "open" for t in self.tasks):
            self.tasks.append({"task": description, "status": "open", "created_at": _now()})

    def close_task(self, description: str, status: str = "done") -> None:
        for t in self.tasks:
            if t["task"] == description and t["status"] == "open":
                t["status"] = status
                t["closed_at"] = _now()

    def open_tasks(self) -> list[str]:
        return [t["task"] for t in self.tasks if t["status"] == "open"]

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def record_test(self, name: str, passed: bool, details: str = "") -> None:
        self.test_results.append({
            "name": name,
            "passed": passed,
            "at": _now(),
            "details": details[:300],
        })
        # keep only last 50
        self.test_results = self.test_results[-50:]

    def last_test_pass_rate(self) -> float:
        recent = self.test_results[-20:]
        if not recent:
            return 1.0
        return sum(1 for t in recent if t["passed"]) / len(recent)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def record_error(self, error: str) -> None:
        self.recent_errors.append(f"{_now()} {error[:200]}")
        self.recent_errors = self.recent_errors[-self._max_errors:]

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def add_decision(self, note: str) -> None:
        self.decisions.append(f"{_now()} {note}")
        self.decisions = self.decisions[-30:]

    # ------------------------------------------------------------------
    # Summary for injection into prompts
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a compact text block to inject into planner/code prompts."""
        lines: list[str] = []

        if self.project_name:
            lines.append(f"PROJECT: {self.project_name}")
        if self.description:
            lines.append(f"DESCRIPTION: {self.description}")

        if self.files:
            lines.append(f"\nEXISTING FILES ({len(self.files)}):")
            for path, meta in list(self.files.items())[-15:]:
                lines.append(f"  {path}  ({meta['size']} chars)  — {meta['summary']}")

        open_t = self.open_tasks()
        if open_t:
            lines.append(f"\nOPEN TASKS ({len(open_t)}):")
            for t in open_t[:10]:
                lines.append(f"  - {t}")

        recent_tests = self.test_results[-10:]
        if recent_tests:
            passed = sum(1 for t in recent_tests if t["passed"])
            lines.append(f"\nRECENT TESTS: {passed}/{len(recent_tests)} passed")
            failed = [t for t in recent_tests if not t["passed"]]
            for t in failed[-3:]:
                lines.append(f"  FAIL: {t['name']} — {t['details'][:100]}")

        if self.decisions:
            lines.append("\nKEY DECISIONS:")
            for d in self.decisions[-5:]:
                lines.append(f"  {d}")

        if self.recent_errors:
            lines.append("\nRECENT ERRORS (last 3):")
            for e in self.recent_errors[-3:]:
                lines.append(f"  {e}")

        if not lines:
            return "(no project context yet)"
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not self.files and not self.tasks and not self.decisions


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _one_line(code: str) -> str:
    """Extract a one-line summary from code (first non-empty, non-comment line)."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", '"""', "'''")):
            return stripped[:80]
    return ""
