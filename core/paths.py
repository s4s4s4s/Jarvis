# core/paths.py
from pathlib import Path

ROOT         = Path(r"C:\jarvis")
LOGS_DIR     = ROOT / "logs"
ROUTER_LOG   = LOGS_DIR / "router.jsonl"
MEMORY_PATH  = ROOT / "data" / "memory.json"
TTS_CHUNKS   = ROOT / "_tts_chunks"  # temp TTS audio chunks


def ensure_dirs() -> None:
    """Create required directories. Called once at startup (app.py)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TTS_CHUNKS.mkdir(parents=True, exist_ok=True)
