# core/paths.py
from pathlib import Path

ROOT = Path(r"C:\jarvis")
LOGS_DIR = ROOT / "logs"
ROUTER_LOG = LOGS_DIR / "router.jsonl"
MEMORY_PATH = ROOT / "data" / "memory.json"


def ensure_dirs() -> None:
    """Create required directories. Call once at application startup (app.py)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
