from pathlib import Path

ROOT = Path(r"C:\jarvis")
LOGS_DIR = ROOT / "logs"
ROUTER_LOG = LOGS_DIR / "router.jsonl"
MEMORY_PATH = ROOT / "data" / "memory.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
# === end of file: core/paths.py ===
