# dev/watch_pass_rate.py
"""
Мониторинг pass rate Jarvis.

Читает logs/self_test.jsonl и logs/self_heal.jsonl,
выводит тренд и алертит если pass_rate < порога.

Запуск:
  python -m dev.watch_pass_rate            # показать последние 10 прогонов
  python -m dev.watch_pass_rate --alert    # exit 1 если pass_rate < 0.90
  python -m dev.watch_pass_rate --watch    # следить в реальном времени (--interval=60)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import LOGS_DIR

ALERT_THRESHOLD = 0.90


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _render(records: list[dict], n: int = 10) -> None:
    last = records[-n:]
    print(f"{'Время':<25} {'Режим':<12} {'Pass':<8} {'Δ':>6}  {'Статус'}")
    print("-" * 65)
    prev_rate = None
    for r in last:
        ts       = r.get("ts", "")[:19]
        mode     = r.get("mode", "—")
        rate     = r.get("pass_rate", 0.0)
        passed   = r.get("passed", 0)
        total    = r.get("total", 0)
        delta_s  = ""
        if prev_rate is not None:
            d = rate - prev_rate
            delta_s = f"{d:+.3f}"
        status = "✅" if rate >= ALERT_THRESHOLD else "🔴 ALERT"
        print(f"{ts:<25} {mode:<12} {passed}/{total} ({rate*100:.1f}%)  {delta_s:>6}  {status}")
        prev_rate = rate


def _load_all_self_tests() -> list[dict]:
    return sorted(
        _load_jsonl(LOGS_DIR / "self_test.jsonl"),
        key=lambda r: r.get("ts", ""),
    )


def check_once(alert_mode: bool = False) -> float:
    records = _load_all_self_tests()
    if not records:
        print("[watch] Нет данных в logs/self_test.jsonl — запусти self_test_agent сначала.")
        return 1.0

    _render(records)

    latest_rate = records[-1].get("pass_rate", 1.0)
    if alert_mode and latest_rate < ALERT_THRESHOLD:
        print(f"\n🔴 ALERT: pass_rate={latest_rate:.3f} < {ALERT_THRESHOLD}")
        alert_path = LOGS_DIR / "alerts.jsonl"
        import datetime
        entry = {
            "ts":         datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "type":       "pass_rate_alert",
            "pass_rate":  latest_rate,
            "threshold":  ALERT_THRESHOLD,
        }
        with open(alert_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return latest_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert",    action="store_true", help="exit 1 если pass_rate < 0.90")
    parser.add_argument("--watch",    action="store_true", help="Непрерывный мониторинг")
    parser.add_argument("--interval", type=int, default=60, help="Интервал --watch в секундах")
    parser.add_argument("--n",        type=int, default=10, help="Показать последние N прогонов")
    args = parser.parse_args()

    if args.watch:
        print(f"[watch] Мониторинг каждые {args.interval}с. Ctrl+C для выхода.")
        try:
            while True:
                print(f"\n{'='*65}")
                rate = check_once(alert_mode=args.alert)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watch] Остановлен.")
            sys.exit(0)
    else:
        rate = check_once(alert_mode=args.alert)
        if args.alert and rate < ALERT_THRESHOLD:
            sys.exit(1)
