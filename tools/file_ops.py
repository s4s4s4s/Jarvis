"""tools/file_ops.py — read/write/list filesystem operations for Jarvis."""
from __future__ import annotations

from pathlib import Path

JARVIS_ROOT = Path("C:/jarvis")


def read_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    text = p.read_text(encoding="utf-8")
    return {"ok": True, "content": text, "lines": len(text.splitlines())}


def write_file(path: str, content: str) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(p), "bytes": len(content.encode())}


def list_dir(path: str = ".") -> dict:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"Directory not found: {path}"}
    entries = [
        {
            "name": e.name,
            "type": "dir" if e.is_dir() else "file",
            "size": e.stat().st_size if e.is_file() else 0,
        }
        for e in sorted(p.iterdir())
    ]
    return {"ok": True, "path": str(p), "entries": entries}
