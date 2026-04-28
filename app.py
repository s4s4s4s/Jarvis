# app.py

from core.paths import ensure_dirs

# Create required directories before any module tries to write to them
ensure_dirs()

# Auto-start Ollama if not running
from core.ollama_guard import ensure_ollama
import sys

if not ensure_ollama():
    sys.exit(1)

from ui.main_window import run

if __name__ == "__main__":
    run()
