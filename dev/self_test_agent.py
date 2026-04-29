# dev/self_test_agent.py
"""
Self-test агент Jarvis.

Прогоняет TEST_CASES через роутер и измеряет pass rate.
Пишет детальный отчёт в logs/self_test.jsonl.

Запуск:
  python -m dev.self_test_agent              # embed + LLM fallback
  python -m dev.self_test_agent --llm-only   # только LLM (baseline)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import LOGS_DIR

logger = logging.getLogger(__name__)

TEST_CASES: list[dict] = [
    # --- weather ---
    {"text": "какая погода сейчас",                "expected_route": "tool",     "expected_tool": "weather"},
    {"text": "погода в Москве",                     "expected_route": "tool",     "expected_tool": "weather"},
    {"text": "идёт ли дождь",                       "expected_route": "tool",     "expected_tool": "weather"},
    {"text": "нужно ли брать зонт",                 "expected_route": "tool",     "expected_tool": "weather"},
    {"text": "прогноз погоды на неделю",             "expected_route": "tool",     "expected_tool": "weather"},
    # --- crypto ---
    {"text": "сколько стоит биткоин",               "expected_route": "tool",     "expected_tool": "crypto.search"},
    {"text": "курс биткоина сейчас",                "expected_route": "tool",     "expected_tool": "crypto.search"},
    {"text": "цена эфира",                          "expected_route": "tool",     "expected_tool": "crypto.search"},
    {"text": "BTC цена",                            "expected_route": "tool",     "expected_tool": "crypto.search"},
    {"text": "биткоин растёт или падает",           "expected_route": "tool",     "expected_tool": "crypto.search"},
    # --- currency ---
    {"text": "курс доллара",                        "expected_route": "tool",     "expected_tool": "currency.rates"},
    {"text": "сколько стоит доллар",                "expected_route": "tool",     "expected_tool": "currency.rates"},
    {"text": "переведи 100 долларов в рубли",        "expected_route": "tool",     "expected_tool": "currency.convert"},
    {"text": "курс евро к рублю",                   "expected_route": "tool",     "expected_tool": "currency.rates"},
    {"text": "покажи курсы валют",                  "expected_route": "tool",     "expected_tool": "currency.rates"},
    # --- timer ---
    {"text": "установи таймер на 10 минут",         "expected_route": "tool",     "expected_tool": "timer.set"},
    {"text": "поставь таймер на 5 минут",           "expected_route": "tool",     "expected_tool": "timer.set"},
    {"text": "напомни через полчаса",               "expected_route": "tool",     "expected_tool": "timer.set"},
    {"text": "какие таймеры активны",               "expected_route": "tool",     "expected_tool": "timer.list"},
    {"text": "отмени таймер",                       "expected_route": "tool",     "expected_tool": "timer.cancel"},
    # --- time ---
    {"text": "сколько сейчас времени",              "expected_route": "tool",     "expected_tool": "time"},
    {"text": "который час",                         "expected_route": "tool",     "expected_tool": "time"},
    {"text": "какое сегодня число",                 "expected_route": "tool",     "expected_tool": "time"},
    {"text": "какой сегодня день недели",           "expected_route": "tool",     "expected_tool": "time"},
    # --- auditor ---
    {"text": "проверь код на баги",                 "expected_route": "tool",     "expected_tool": "auditor.run"},
    {"text": "проаудируй проект",                   "expected_route": "tool",     "expected_tool": "auditor.run"},
    # --- chat ---
    {"text": "привет",                              "expected_route": "chat",     "expected_tool": None},
    {"text": "как дела",                            "expected_route": "chat",     "expected_tool": None},
    {"text": "что такое нейронная сеть",            "expected_route": "chat",     "expected_tool": None},
    {"text": "как работает трансформер в LLM",      "expected_route": "chat",     "expected_tool": None},
    {"text": "что такое top-p в языковых моделях",  "expected_route": "chat",     "expected_tool": None},
    {"text": "что такое контекстное окно модели",   "expected_route": "chat",     "expected_tool": None},
    # --- ОМОНИМЫ ---
    {"text": "какую температуру модели использовать",     "expected_route": "chat", "expected_tool": None},
    {"text": "что означает параметр temperature у LLM",   "expected_route": "chat", "expected_tool": None},
    {"text": "при какой температуре тела вызывать врача", "expected_route": "chat", "expected_tool": None},
    {"text": "температура кипения воды",                  "expected_route": "chat", "expected_tool": None},
    {"text": "курс обучения по машинному обучению",        "expected_route": "chat", "expected_tool": None},
    {"text": "курс истории России",                       "expected_route": "chat", "expected_tool": None},
    {"text": "скорость передачи данных в нейросети",       "expected_route": "chat", "expected_tool": None},
    {"text": "время обучения модели",                      "expected_route": "chat", "expected_tool": None},
    # --- web ---
    {"text": "найди последние новости про OpenAI",   "expected_route": "web",     "expected_tool": None},
    {"text": "что нового в мире технологий",         "expected_route": "web",     "expected_tool": None},
    {"text": "что произошло сегодня в России",       "expected_route": "web",     "expected_tool": None},
    {"text": "новости про ChatGPT",                  "expected_route": "web",     "expected_tool": None},
    # --- deep ---
    {"text": "проанализируй плюсы и минусы трансформеров", "expected_route": "deep", "expected_tool": None},
    {"text": "сделай подробный разбор технологии RAG",     "expected_route": "deep", "expected_tool": None},
    {"text": "объясни детально как устроен RLHF",          "expected_route": "deep", "expected_tool": None},
    # --- memory ---
    {"text": "что ты обо мне знаешь",               "expected_route": "memory",   "expected_tool": None},
    {"text": "что ты помнишь про меня",             "expected_route": "memory",   "expected_tool": None},
    {"text": "ты знаешь как меня зовут",            "expected_route": "memory",   "expected_tool": None},
    # --- feedback ---
    {"text": "ты ошибся",  "expected_route": "feedback", "expected_tool": "feedback.wrong"},
    {"text": "всё верно",  "expected_route": "feedback", "expected_tool": "feedback.correct"},
    {"text": "молодец",    "expected_route": "feedback", "expected_tool": "feedback.correct"},
]


def _build_router(use_embed: bool):
    """
    Собираем функцию маршрутизации.
    Fixed: раньше импортировалось несуществующее brain.ask._route — теперь _route_llm.
    """
    from brain.ask import _route_llm

    if not use_embed:
        def _only_llm(text: str) -> dict:
            r = _route_llm(text)
            r.setdefault("_source", "llm")
            return r
        return _only_llm

    from brain.router_embed import route_embed

    def _route_smart(text: str) -> dict:
        result = route_embed(text)
        if result is not None:
            return result
        r = _route_llm(text)
        r.setdefault("_source", "llm")
        return r
    return _route_smart


def run_self_test(use_embed: bool = True) -> dict:
    router = _build_router(use_embed)

    results: list[dict] = []
    for case in TEST_CASES:
        try:
            decision = router(case["text"])
        except Exception as e:
            logger.error(f"[self_test] router error for '{case['text']}': {e}")
            decision = {"route": "error", "tool": None, "confidence": 0.0, "_source": "error"}

        correct = (
            decision.get("route") == case["expected_route"]
            and decision.get("tool") == case["expected_tool"]
        )
        results.append({
            **case,
            "got_route":  decision.get("route"),
            "got_tool":   decision.get("tool"),
            "confidence": decision.get("confidence", 0.0),
            "source":     decision.get("_source", "unknown"),
            "correct":    correct,
        })

    passed = sum(1 for r in results if r["correct"])
    total  = len(results)
    failed = [r for r in results if not r["correct"]]

    by_source: dict[str, dict[str, int]] = {}
    for r in results:
        src = r["source"]
        bucket = by_source.setdefault(src, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r["correct"]:
            bucket["passed"] += 1

    summary = {
        "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode":       "embed+llm" if use_embed else "llm_only",
        "passed":     passed,
        "total":      total,
        "pass_rate":  round(passed / total, 3),
        "by_source":  by_source,
        "failed_cases": [
            {
                "text":       r["text"],
                "expected":   {"route": r["expected_route"], "tool": r["expected_tool"]},
                "got":        {"route": r["got_route"],      "tool": r["got_tool"]},
                "confidence": r["confidence"],
                "source":     r["source"],
            }
            for r in failed
        ],
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "self_test.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return {**summary, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-only", action="store_true", help="Тестировать только LLM-роутер")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Запуск self-test ({'llm_only' if args.llm_only else 'embed+llm'})...")
    report = run_self_test(use_embed=not args.llm_only)

    print(f"\nРезультат: {report['passed']}/{report['total']} ({report['pass_rate']*100:.1f}%)")
    print("По источникам:")
    for src, b in report["by_source"].items():
        pr = (b["passed"] / b["total"] * 100) if b["total"] else 0.0
        print(f"  {src:10s}  {b['passed']}/{b['total']}  ({pr:.1f}%)")

    if report["failed"]:
        print("\nПровалившиеся кейсы:")
        for r in report["failed"]:
            print(f"  '{r['text']}'")
            print(f"    ожидалось: route={r['expected_route']} tool={r['expected_tool']}")
            print(f"    получено:  route={r['got_route']} tool={r['got_tool']} "
                  f"(conf={r['confidence']:.2f}, src={r['source']})")

    sys.exit(0 if report["pass_rate"] >= 0.90 else 1)
