# core/paths.py
import os
from pathlib import Path

# ROOT можно переопределить через переменную окружения JARVIS_ROOT
ROOT         = Path(os.environ.get("JARVIS_ROOT", r"C:\jarvis"))
LOGS_DIR     = ROOT / "logs"
ROUTER_LOG   = LOGS_DIR / "router.jsonl"
MEMORY_PATH  = ROOT / "data" / "memory.json"
TTS_CHUNKS   = ROOT / "_tts_chunks"  # temp TTS audio chunks

# FIX (audit 3): ассеты теперь тоже идут через ROOT — раньше они были
# захардкожены в core/config.py (_ROOT = Path(r"C:\jarvis")), и при смене
# JARVIS_ROOT звуки продолжали искаться в C:\jarvis\assets.
ASSETS_DIR            = ROOT / "assets"
REFERENCE_WAV         = str(ASSETS_DIR / "reference.wav")
ACTIVATE_SOUND_PATH   = str(ASSETS_DIR / "activate.wav")
DEACTIVATE_SOUND_PATH = str(ASSETS_DIR / "deactivate.wav")


def ensure_dirs() -> None:
    """Create required directories. Called once at startup (app.py)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TTS_CHUNKS.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
