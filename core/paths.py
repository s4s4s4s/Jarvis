import os
from pathlib import Path

# ROOT можно переопределить через переменную среды JARVIS_ROOT
ROOT         = Path(os.environ.get("JARVIS_ROOT", r"C:\jarvis"))
LOGS_DIR     = ROOT / "logs"
ROUTER_LOG   = LOGS_DIR / "router.jsonl"
MEMORY_PATH  = ROOT / "data" / "memory.json"   # legacy flat-JSON (migration source)
TTS_CHUNKS   = ROOT / "_tts_chunks"

ASSETS_DIR            = ROOT / "assets"
REFERENCE_WAV         = str(ASSETS_DIR / "reference.wav")
ACTIVATE_SOUND_PATH   = str(ASSETS_DIR / "activate.wav")
DEACTIVATE_SOUND_PATH = str(ASSETS_DIR / "deactivate.wav")

# ─── Self-learning ──────────────────────────────────────────────────────────────────────────────
ROUTE_EXAMPLES   = ROOT / "data" / "route_examples.jsonl"
FEEDBACK_LOG     = LOGS_DIR / "feedback.jsonl"
LEARNING_REPORT  = LOGS_DIR / "learning_report.jsonl"
FEEDBACK_ARCHIVE = LOGS_DIR / "feedback_archive.jsonl"

# ─── Векторная память ─────────────────────────────────────────────────────────────────────────────
CHROMA_DIR = ROOT / "data" / "chroma_memory"


def ensure_dirs() -> None:
    """Create required directories. Called once at startup (app.py)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TTS_CHUNKS.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
