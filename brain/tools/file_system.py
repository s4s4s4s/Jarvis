"""
brain/tools/file_system.py
File system access tool for Jarvis agents.
Provides read/write/list/mkdir operations with safety checks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Default output directory — all agent-generated files go here
DEFAULT_OUTPUT_DIR = Path("output")


class FileSystemTool:
    """
    Safe file system operations for Jarvis agents.
    All writes are scoped to output_dir by default.
    """

    def __init__(self, output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_file(self, filename: str, content: str) -> Path:
        """Write content to output_dir/filename. Creates subdirs if needed."""
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("[FS] Wrote %d bytes to %s", len(content), path)
        return path

    def read_file(self, path: str | Path) -> str:
        """Read file contents."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_text(encoding="utf-8")

    def list_dir(self, subdir: str = "") -> list[str]:
        """List files in output_dir (or a subdirectory of it)."""
        target = self.output_dir / subdir if subdir else self.output_dir
        if not target.exists():
            return []
        return [str(p.relative_to(self.output_dir)) for p in sorted(target.rglob("*")) if p.is_file()]

    def make_dir(self, dirname: str) -> Path:
        """Create a directory inside output_dir."""
        path = self.output_dir / dirname
        path.mkdir(parents=True, exist_ok=True)
        logger.info("[FS] Created dir %s", path)
        return path


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract code blocks from markdown text.
    Returns list of (language, code) tuples.
    Handles: ```python\n...``` and plain indented blocks.
    """
    pattern = re.compile(r"```([\w]*)\n([\s\S]*?)```", re.MULTILINE)
    blocks = []
    for match in pattern.finditer(text):
        lang = match.group(1).strip() or "txt"
        code = match.group(2)
        blocks.append((lang, code))
    return blocks


def suggest_filename(goal: str, lang: str = "py") -> str:
    """
    Derive a safe snake_case filename from a task goal string.
    E.g. 'Write BTC price monitor script' -> 'btc_price_monitor.py'
    """
    # Strip common noise words
    noise = {"write", "create", "implement", "add", "build", "make",
             "develop", "generate", "the", "a", "an", "for", "to",
             "with", "and", "in", "of", "script", "code", "function"}
    words = re.sub(r"[^\w\s]", "", goal.lower()).split()
    meaningful = [w for w in words if w not in noise][:6]
    name = "_".join(meaningful) if meaningful else "output"
    ext = {"python": "py", "py": "py", "javascript": "js", "js": "js",
           "bash": "sh", "sh": "sh", "txt": "txt"}.get(lang, lang or "py")
    return f"{name}.{ext}"
