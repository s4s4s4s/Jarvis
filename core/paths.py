import os
from pathlib import Path

# fix #15: ROOT определяется относительно этого файла как fallback,
# а не захардкоженным C:\jarvis. Если JARVIS_ROOT выставлен — используем его.
# Иначе берём два уровня вверх от core/paths.py (т.е. корень репозитория).
_default_root = Path(__file__).resolve().parent.parent
ROOT         = Path(os.environ.get("JARVIS_ROOT", str(_default_root)))

LOGS_DIR     = ROOT / "logs"
ROUTER_LOG   = LOGS_DIR / "router.jsonl"
MEMORY_PATH  = ROOT / "data" / "memory.json"   # legacy flat-JSON (migration source)
TTS_CHUNKS   = ROOT / "_tts_chunks"

ASSETS_DIR            = ROOT / "assets"
REFERENCE_WAV         = str(ASSETS_DIR / "reference.wav")
ACTIVATE_SOUND_PATH   = str(ASSETS_DIR / "activate.wav")
DEACTIVATE_SOUND_PATH = str(ASSETS_DIR / "deactivate.wav")

# Self-learning
ROUTE_EXAMPLES   = ROOT / "data" / "route_examples.jsonl"
FEEDBACK_LOG     = LOGS_DIR / "feedback.jsonl"
LEARNING_REPORT  = LOGS_DIR / "learning_report.jsonl"
FEEDBACK_ARCHIVE = LOGS_DIR / "feedback_archive.jsonl"

# Векторная память (ChromaDB)
CHROMA_DIR = ROOT / "data" / "chroma_memory"

# Level 4: проекты ProjectAgent
PROJECTS_DIR = ROOT / "data" / "projects"
PROJECT_LOG  = LOGS_DIR / "projects.jsonl"


def ensure_dirs() -> None:
    """Create required directories. Called once at startup (app.py)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TTS_CHUNKS.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
