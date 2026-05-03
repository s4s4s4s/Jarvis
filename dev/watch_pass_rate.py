# dev/watch_pass_rate.py
"""
Мониторинг pass rate Jarvis.

Читает logs/self_test.jsonl и logs/self_heal.jsonl,
выводит тренд и алертит если pass_rate < порога.

Запуск:
  python -m dev.watch_pass_rate            # показать последние 10 прогонов
  python -m dev.watch_pass_rate --alert    # exit 1 если pass_rate < 0.90
  python -m dev.watch_pass_rate --watch    # следить в реальном времени (--interval=60)

fix W1:
  - _render() теперь принимает n= и уважает args.n (раньше --n игнорировался)
  - в --watch + --alert режиме exit 1 происходит как ожидается,
    а не тихо глотается KeyboardInterrupt-путём
"""
from __future__ import annotations

import argparse
import datetime
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
    # fix W1: раньше параметр n передавался, но внутри был захардкожен [-10:] без n
    last = records[-n:]
    print(f"{'\u0412\u0440\u0435\u043c\u044f':<25} {'\u0420\u0435\u0436\u0438\u043c':<12} {'Pass':<8} {'\u0394':>6}  {'\u0421\u0442\u0430\u0442\u0443\u0441'}")
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
        status = "\u2705" if rate >= ALERT_THRESHOLD else "\U0001f534 ALERT"
        print(f"{ts:<25} {mode:<12} {passed}/{total} ({rate*100:.1f}%)  {delta_s:>6}  {status}")
        prev_rate = rate


def _load_all_self_tests() -> list[dict]:
    return sorted(
        _load_jsonl(LOGS_DIR / "self_test.jsonl"),
        key=lambda r: r.get("ts", ""),
    )


def _write_alert(latest_rate: float) -> None:
    alert_path = LOGS_DIR / "alerts.jsonl"
    entry = {
        "ts":        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "type":      "pass_rate_alert",
        "pass_rate": latest_rate,
        "threshold": ALERT_THRESHOLD,
    }
    with open(alert_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def check_once(alert_mode: bool = False, n: int = 10) -> float:
    # fix W1: передаём n в _render()
    records = _load_all_self_tests()
    if not records:
        print("[watch] Нет данных в logs/self_test.jsonl — запусти self_test_agent сначала.")
        return 1.0

    _render(records, n=n)

    latest_rate = records[-1].get("pass_rate", 1.0)
    if alert_mode and latest_rate < ALERT_THRESHOLD:
        print(f"\n\U0001f534 ALERT: pass_rate={latest_rate:.3f} < {ALERT_THRESHOLD}")
        _write_alert(latest_rate)

    return latest_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert",    action="store_true", help="exit 1 если pass_rate < 0.90")
    parser.add_argument("--watch",    action="store_true", help="Непрерывный мониторинг")
    parser.add_argument("--interval", type=int, default=60, help="Интервал --watch в секундах")
    parser.add_argument("--n",        type=int, default=10, help="Показать последние N прогонов")
    args = parser.parse_args()

    # fix W1: отслеживаем алерт в --watch режиме + exit 1 после KeyboardInterrupt
    alert_triggered = False

    if args.watch:
        print(f"[watch] Мониторинг каждые {args.interval}\u0441. Ctrl+C для выхода.")
        try:
            while True:
                print(f"\n{'='*65}")
                rate = check_once(alert_mode=args.alert, n=args.n)
                if args.alert and rate < ALERT_THRESHOLD:
                    alert_triggered = True
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watch] Остановлен.")
        sys.exit(1 if alert_triggered else 0)
    else:
        rate = check_once(alert_mode=args.alert, n=args.n)
        if args.alert and rate < ALERT_THRESHOLD:
            sys.exit(1)
