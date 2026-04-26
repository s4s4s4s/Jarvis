# app.py

from core.paths import ensure_dirs

# Create required directories before any module tries to write to them
ensure_dirs()

from ui.main_window import run

if __name__ == "__main__":
    run()
