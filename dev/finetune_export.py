"""
dev/finetune_export.py — экспорт логов для fine-tuning через unsloth.

Отбирает записи из feedback.jsonl / feedback_archive.jsonl со статусом verified_success
или auto_success (уверенная маршрутизация) и формирует Alpaca-совместимый JSONL
для unsloth FastLanguageModel.from_pretrained + SFTTrainer.

Формат каждой строки выходного JSONL:
  {"instruction": "<запрос>", "input": "", "output": "<размещенный ответ>"}

Запуск:
  python -m dev.finetune_export
  python -m dev.finetune_export --min-records 200 --out data/finetune_v2.jsonl
  python -m dev.finetune_export --stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

from core.paths import LOGS_DIR, ROOT

DEFAULT_OUT    = ROOT / "data" / "finetune_train.jsonl"
SOURCE_FILES   = [
    LOGS_DIR / "feedback.jsonl",
    LOGS_DIR / "feedback_archive.jsonl",
]
GOOD_OUTCOMES  = {"verified_success", "auto_success"}
MIN_TEXT_LEN   = 8
MIN_ANSWER_LEN = 10


def _load_records() -> list[dict]:
    records: list[dict] = []
    for path in SOURCE_FILES:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _to_alpaca(record: dict) -> dict | None:
    """Преобразует запись feedback → Alpaca-формат."""
    text   = (record.get("text")   or "").strip()
    answer = (record.get("answer") or "").strip()
    outcome = record.get("outcome", "")

    if outcome not in GOOD_OUTCOMES:
        return None
    if len(text) < MIN_TEXT_LEN or len(answer) < MIN_ANSWER_LEN:
        return None
    # Исключаем слишком короткие однословные ответы
    if len(answer.split()) < 3:
        return None

    return {
        "instruction": text,
        "input":       "",
        "output":      answer,
        # мета для отладки (не используется тренировкой)
        "_meta": {
            "route":   record.get("route"),
            "outcome": outcome,
            "ts":      record.get("ts", ""),
        },
    }


def export(
    out_path: Path = DEFAULT_OUT,
    min_records: int = 0,
    verbose: bool = True,
) -> int:
    """
    Экспортирует обучающие записи в JSONL.
    Возвращает количество записей.
    """
    records  = _load_records()
    samples  = [s for r in records if (s := _to_alpaca(r)) is not None]

    # дедупликация по (instruction, output)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for s in samples:
        key = (s["instruction"], s["output"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    if verbose:
        print(f"\u0412сего записей feedback:  {len(records)}")
        print(f"Подходящих (вериф.усп.): {len(samples)}")
        print(f"Уникальных:           {len(unique)}")

        route_counts = Counter(s["_meta"]["route"] for s in unique)
        print("\u0420аспределение по route:")
        for route, cnt in sorted(route_counts.items(), key=lambda x: -x[1]):
            print(f"  {route:20s} {cnt}")

    if len(unique) < min_records:
        print(f"Недостаточно записей: {len(unique)} < {min_records}. Не сохраняю.")
        return len(unique)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in unique:
            # убираем _meta перед записью
            row = {k: v for k, v in s.items() if k != "_meta"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\u0421охранено в {out_path} ({len(unique)} записей)")
    return len(unique)


def print_stats() -> None:
    records = _load_records()
    outcomes = Counter(r.get("outcome", "нет") for r in records)
    routes   = Counter(r.get("route", "нет") for r in records)
    print(f"Итого записей: {len(records)}")
    print("\nOutcome:")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v}")
    print("\nRoute:")
    for k, v in sorted(routes.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")
    # тренд по дням
    from collections import defaultdict
    daily: dict = defaultdict(int)
    for r in records:
        day = (r.get("ts") or "")[:10]
        if day:
            daily[day] += 1
    if daily:
        print("\nТренд (последние 10 дней):")
        for day in sorted(daily)[-10:]:
            bar = "█" * min(daily[day], 40)
            print(f"  {day} {bar} {daily[day]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis экспорт для fine-tuning")
    parser.add_argument("--out",         default=str(DEFAULT_OUT),
                        help="Путь выходного JSONL")
    parser.add_argument("--min-records", type=int, default=0,
                        help="Мин. количество записей (если меньше — не сохранять)")
    parser.add_argument("--stats",       action="store_true",
                        help="Показать статистику без экспорта")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        export(
            out_path=Path(args.out),
            min_records=args.min_records,
        )


if __name__ == "__main__":
    main()
