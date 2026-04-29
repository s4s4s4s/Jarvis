# dev/nightly_self_heal.py
"""
Nightly self-heal цикл Jarvis.

Что делает:
  1. Бэкапит data/route_examples.jsonl
  2. Запускает self_test (baseline)
  3. Запускает learning_loop.run_learning_cycle()
  4. Запускает self_test (ещё раз post)
  5. Если pass_rate упал ≥ REGRESSION_TOLERANCE —откатывает бэкап
  6. Сбрасывает embed-кэш, пишет отчёт в logs/self_heal.jsonl

Запуск (например из Task Scheduler или ручно):
  python -m dev.nightly_self_heal
Выходной код:
  0 — всё ок, обучение принято
  1 — был откат (регрессия)
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import ROUTE_EXAMPLES, LOGS_DIR

logger = logging.getLogger(__name__)

# Максимальное допустимое падение pass rate после цикла
REGRESSION_TOLERANCE = 0.02   # 2 процентных пункта
# Абсолютный порог — ниже всегда откат
MIN_PASS_RATE        = 0.85


def _backup() -> Path | None:
    """Copy route_examples.jsonl to .bak.<ts>.jsonl."""
    if not ROUTE_EXAMPLES.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = ROUTE_EXAMPLES.with_suffix(f".bak.{ts}.jsonl")
    shutil.copy2(ROUTE_EXAMPLES, backup)
    logger.info(f"[self_heal] Бэкап сохранён: {backup.name}")
    return backup


def _restore(backup: Path | None) -> bool:
    """Restore from backup and invalidate embed cache."""
    if backup is None or not backup.exists():
        logger.error("[self_heal] Бэкап не найден, откат невозможен")
        return False
    shutil.copy2(backup, ROUTE_EXAMPLES)
    try:
        from brain.router_embed import invalidate_cache
        invalidate_cache()
    except Exception as e:
        logger.error(f"[self_heal] Ошибка сброса кэша после отката: {e}")
    logger.warning(f"[self_heal] Откат route_examples.jsonl ← {backup.name}")
    return True


def _write_report(report: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "self_heal.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")


def run() -> dict:
    """
    Полный self-heal цикл.
    Возвращает детальный отчёт (сериализуется в self_heal.jsonl).
    """
    from dev.self_test_agent import run_self_test
    from brain.learning_loop import run_learning_cycle

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report: dict = {"ts": ts, "rolled_back": False}

    # ── 1. Baseline self-test ─────────────────────────────────────────────────
    logger.info("[self_heal] Шаг 1: baseline self-test...")
    try:
        baseline = run_self_test(use_embed=True)
    except Exception as e:
        logger.error(f"[self_heal] baseline сбой: {e}")
        report["error"] = f"baseline failed: {e}"
        _write_report(report)
        return report

    report["baseline"] = {
        "pass_rate": baseline["pass_rate"],
        "passed":    baseline["passed"],
        "total":     baseline["total"],
    }
    logger.info(f"[self_heal] baseline pass_rate={baseline['pass_rate']:.3f}")

    # ── 2. Бэкап ────────────────────────────────────────────────────────────────
    backup = _backup()

    # ── 3. Learning cycle ────────────────────────────────────────────────────
    logger.info("[self_heal] Шаг 2: learning cycle...")
    try:
        learning = run_learning_cycle()
        report["learning"] = {
            "examples_added":   learning["examples_added"],
            "examples_removed": learning["examples_removed"],
            "total_examples":   learning["total_examples"],
            "verified_success": learning["verified_success"],
            "verified_failure":  learning["verified_failure"],
            "auto_success_added": learning["auto_success_added"],
        }
    except Exception as e:
        logger.error(f"[self_heal] learning cycle сбой: {e}")
        report["learning_error"] = str(e)
        report["rolled_back"] = _restore(backup)
        _write_report(report)
        return report

    # ── 4. Post self-test ────────────────────────────────────────────────────
    logger.info("[self_heal] Шаг 3: post self-test...")
    try:
        post = run_self_test(use_embed=True)
    except Exception as e:
        logger.error(f"[self_heal] post self-test сбой: {e}")
        report["post_error"] = str(e)
        report["rolled_back"] = _restore(backup)
        _write_report(report)
        return report

    report["post"] = {
        "pass_rate": post["pass_rate"],
        "passed":    post["passed"],
        "total":     post["total"],
    }
    delta = post["pass_rate"] - baseline["pass_rate"]
    report["delta"] = round(delta, 3)
    logger.info(f"[self_heal] post pass_rate={post['pass_rate']:.3f}  delta={delta:+.3f}")

    # ── 5. Откат если регрессия ────────────────────────────────────────────
    must_rollback = (
        post["pass_rate"] < MIN_PASS_RATE
        or delta < -REGRESSION_TOLERANCE
    )
    if must_rollback:
        logger.warning(
            f"[self_heal] Регрессия! post={post['pass_rate']:.3f} "
            f"delta={delta:+.3f} → откат..."
        )
        report["rolled_back"] = _restore(backup)
        # Верификация отката
        try:
            verify = run_self_test(use_embed=True)
            report["verify"] = {
                "pass_rate": verify["pass_rate"],
                "passed":    verify["passed"],
            }
            logger.info(f"[self_heal] После отката pass_rate={verify['pass_rate']:.3f}")
        except Exception as e:
            logger.error(f"[self_heal] verify сбой: {e}")
            report["verify_error"] = str(e)
    else:
        logger.info("[self_heal] Обучение принято — откат не нужен.")

    _write_report(report)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if not result.get("rolled_back") else 1)
