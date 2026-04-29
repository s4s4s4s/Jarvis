"""
self_test_planner.py — тест-сьют для PlannerAgent (brain/agents/planner.py)

Запуск:
    python -m dev.self_test_planner
    python -m dev.self_test_planner --verbose

Тесты охватывают:
  - валидацию JSON-схемы плана (_validate_plan)
  - безопасную подстановку результатов (_safe_substitute)
  - strip markdown (_strip_markdown)
  - мок-вызов run() с подменённым chat + call_tool
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# ── импорты тестируемых внутренних функций ───────────────────────────────────
from brain.agents.planner import (
    _validate_plan,
    _safe_substitute,
    _strip_markdown,
    _make_plan,
    run,
    MAX_STEPS,
    STEP_TIMEOUT,
)
from tools.registry import ToolResult


# ═══════════════════════════════════════════════════════════════════════════════
# Вспомогательные фабрики
# ═══════════════════════════════════════════════════════════════════════════════

def _plan(*steps: dict) -> list[dict]:
    """Собирает список шагов с автоматическим answer на конце если нет."""
    steps_list = list(steps)
    if not steps_list or steps_list[-1].get("type") != "answer":
        steps_list.append({"step": len(steps_list) + 1, "type": "answer", "description": "синтез"})
    return steps_list


def _tool_step(n: int, tool: str, args: dict | None = None) -> dict:
    return {"step": n, "type": "tool", "tool": tool, "args": args or {}, "description": f"шаг {n}"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Тесты _strip_markdown
# ═══════════════════════════════════════════════════════════════════════════════

class TestStripMarkdown(unittest.TestCase):

    def test_clean_json_unchanged(self):
        raw = '{"plan": []}'
        self.assertEqual(_strip_markdown(raw), raw)

    def test_strips_backtick_json(self):
        raw = '```json\n{"plan": []}\n```'
        self.assertEqual(_strip_markdown(raw), '{"plan": []}')

    def test_strips_plain_backticks(self):
        raw = '```\n{"plan": []}\n```'
        self.assertEqual(_strip_markdown(raw), '{"plan": []}')

    def test_strips_whitespace(self):
        raw = '  {"plan": []}  '
        self.assertEqual(_strip_markdown(raw), '{"plan": []}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Тесты _validate_plan
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidatePlan(unittest.TestCase):

    # --- позитивные ---
    def test_valid_single_tool_and_answer(self):
        plan = _plan(_tool_step(1, "time"))
        ok, reason = _validate_plan(plan)
        self.assertTrue(ok, reason)

    def test_valid_two_tools_and_answer(self):
        plan = _plan(
            _tool_step(1, "time"),
            _tool_step(2, "weather", {"location": "Moscow"}),
        )
        ok, reason = _validate_plan(plan)
        self.assertTrue(ok, reason)

    def test_valid_plan_with_substitution_in_args(self):
        plan = _plan(
            _tool_step(1, "file.read", {"path": "~/a.txt"}),
            _tool_step(2, "file.write", {"path": "~/b.txt", "content": "{step1_result}"}),
        )
        ok, reason = _validate_plan(plan)
        self.assertTrue(ok, reason)

    # --- негативные ---
    def test_empty_plan(self):
        ok, reason = _validate_plan([])
        self.assertFalse(ok)

    def test_plan_not_list(self):
        ok, reason = _validate_plan({})  # type: ignore
        self.assertFalse(ok)

    def test_last_step_not_answer(self):
        plan = [_tool_step(1, "time"), _tool_step(2, "weather")]
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)
        self.assertIn("answer", reason)

    def test_duplicate_step_numbers(self):
        plan = [
            _tool_step(1, "time"),
            _tool_step(1, "weather"),
            {"step": 2, "type": "answer", "description": "x"},
        ]
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)
        self.assertIn("дублирующийся", reason)

    def test_unknown_tool(self):
        plan = _plan(_tool_step(1, "nonexistent.tool"))
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)
        self.assertIn("не существует", reason)

    def test_too_many_steps(self):
        steps = [_tool_step(i, "time") for i in range(1, MAX_STEPS + 2)]
        steps.append({"step": MAX_STEPS + 2, "type": "answer", "description": "x"})
        ok, reason = _validate_plan(steps)
        self.assertFalse(ok)
        self.assertIn("много", reason)

    def test_unknown_step_type(self):
        plan = [
            {"step": 1, "type": "magic", "tool": "time", "args": {}, "description": "x"},
            {"step": 2, "type": "answer", "description": "x"},
        ]
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)
        self.assertIn("type", reason)

    def test_tool_step_missing_tool_name(self):
        plan = [
            {"step": 1, "type": "tool", "args": {}, "description": "x"},
            {"step": 2, "type": "answer", "description": "x"},
        ]
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)

    def test_args_not_dict(self):
        plan = [
            {"step": 1, "type": "tool", "tool": "time", "args": "bad", "description": "x"},
            {"step": 2, "type": "answer", "description": "x"},
        ]
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)

    def test_step_number_not_int(self):
        plan = [
            {"step": "one", "type": "tool", "tool": "time", "args": {}, "description": "x"},
            {"step": 2, "type": "answer", "description": "x"},
        ]
        ok, reason = _validate_plan(plan)
        self.assertFalse(ok)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Тесты _safe_substitute
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeSubstitute(unittest.TestCase):

    def test_simple_substitution(self):
        args = {"content": "{step1_result}"}
        result = _safe_substitute(args, {1: "hello"})
        self.assertEqual(result["content"], "hello")

    def test_missing_step_keeps_placeholder(self):
        """Если шаг ещё не выполнен — placeholder остаётся как есть."""
        args = {"content": "{step3_result}"}
        result = _safe_substitute(args, {1: "data"})
        self.assertEqual(result["content"], "{step3_result}")

    def test_truncates_long_result(self):
        long_val = "x" * 5000
        args = {"content": "{step1_result}"}
        result = _safe_substitute(args, {1: long_val})
        self.assertLessEqual(len(result["content"]), 2001)

    def test_non_string_result_serialized(self):
        args = {"content": "{step1_result}"}
        result = _safe_substitute(args, {1: {"key": "value"}})
        self.assertIn("key", result["content"])

    def test_non_string_arg_unchanged(self):
        args = {"count": 42}
        result = _safe_substitute(args, {1: "irrelevant"})
        self.assertEqual(result["count"], 42)

    def test_multiple_substitutions_in_one_string(self):
        args = {"msg": "a={step1_result} b={step2_result}"}
        result = _safe_substitute(args, {1: "AAA", 2: "BBB"})
        self.assertEqual(result["msg"], "a=AAA b=BBB")

    def test_empty_args(self):
        result = _safe_substitute({}, {1: "x"})
        self.assertEqual(result, {})


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Тест run() — мок LLM + мок tools
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlannerRun(unittest.TestCase):

    def _mock_plan_json(self, steps: list[dict]) -> str:
        return json.dumps({"plan": steps})

    def test_run_simple_tool_then_answer(self):
        """Один шаг tool + answer → должен вернуть строку."""
        plan_steps = [
            _tool_step(1, "time"),
            {"step": 2, "type": "answer", "description": "финал"},
        ]
        plan_json = self._mock_plan_json(plan_steps)

        call_count = [0]
        def fake_chat(model, msgs, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_json   # _make_plan
            return "Сейчас 10:00"   # синтез

        fake_tool_result = ToolResult(ok=True, data="10:00:00")
        with patch("brain.agents.planner.chat", side_effect=fake_chat), \
             patch("brain.agents.planner.call_tool", return_value=fake_tool_result):
            answer = run("который час?", [])
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

    def test_run_tool_failure_still_returns_string(self):
        """Если tool падает — run не должен бросать исключение."""
        plan_steps = [
            _tool_step(1, "file.read", {"path": "~/missing.txt"}),
            {"step": 2, "type": "answer", "description": "финал"},
        ]
        plan_json = self._mock_plan_json(plan_steps)
        call_count = [0]
        def fake_chat(model, msgs, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_json
            return "Файл не найден"

        broken_result = ToolResult(ok=False, error="File not found")
        with patch("brain.agents.planner.chat", side_effect=fake_chat), \
             patch("brain.agents.planner.call_tool", return_value=broken_result):
            answer = run("прочитай файл", [])
        self.assertIsInstance(answer, str)

    def test_run_invalid_plan_returns_fallback(self):
        """Если LLM возвращает мусор оба раза — run возвращает осмысленный fallback."""
        with patch("brain.agents.planner.chat", return_value="not json at all"):
            answer = run("сделай что-нибудь", [])
        self.assertIsInstance(answer, str)
        self.assertIn("план", answer.lower())

    def test_run_timeout_step_continues(self):
        """Timeout на шаге не должен останавливать выполнение остальных шагов."""
        plan_steps = [
            _tool_step(1, "time"),
            _tool_step(2, "weather", {"location": "Moscow"}),
            {"step": 3, "type": "answer", "description": "финал"},
        ]
        plan_json = self._mock_plan_json(plan_steps)
        call_count = [0]
        def fake_chat(model, msgs, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_json
            return "выполнено"

        import time
        invocations = [0]
        def slow_then_fast(name, args):
            invocations[0] += 1
            if invocations[0] == 1:
                time.sleep(STEP_TIMEOUT + 5)  # симулируем зависание
            return ToolResult(ok=True, data="ok")

        with patch("brain.agents.planner.chat", side_effect=fake_chat), \
             patch("brain.agents.planner.call_tool", side_effect=slow_then_fast):
            # уменьшаем timeout в тесте чтобы не ждать 30 сек
            import brain.agents.planner as planner_mod
            original_timeout = planner_mod.STEP_TIMEOUT
            planner_mod.STEP_TIMEOUT = 0.3
            try:
                answer = run("время и погода", [])
            finally:
                planner_mod.STEP_TIMEOUT = original_timeout
        self.assertIsInstance(answer, str)

    def test_run_substitution_used_between_steps(self):
        """Результат шага 1 подставляется в args шага 2."""
        plan_steps = [
            _tool_step(1, "file.read", {"path": "~/in.txt"}),
            _tool_step(2, "file.write", {"path": "~/out.txt", "content": "{step1_result}"}),
            {"step": 3, "type": "answer", "description": "финал"},
        ]
        plan_json = json.dumps({"plan": plan_steps})
        call_count = [0]
        written_content: list = []

        def fake_chat(model, msgs, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_json
            return "записано"

        def fake_tool(name, args):
            if name == "file.read":
                return ToolResult(ok=True, data="CONTENT_FROM_FILE")
            if name == "file.write":
                written_content.append(args.get("content", ""))
                return ToolResult(ok=True, data="written")
            return ToolResult(ok=False, error="unknown")

        with patch("brain.agents.planner.chat", side_effect=fake_chat), \
             patch("brain.agents.planner.call_tool", side_effect=fake_tool):
            run("прочитай in.txt и запиши в out.txt", [])

        self.assertEqual(len(written_content), 1)
        self.assertEqual(written_content[0], "CONTENT_FROM_FILE")


# ═══════════════════════════════════════════════════════════════════════════════
# Запуск
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PlannerAgent test suite")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result  = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    rate   = passed / total * 100 if total else 0

    print(f"\n{'='*50}")
    print(f"PlannerAgent self-test: {passed}/{total} passed ({rate:.0f}%)")
    if result.failures or result.errors:
        print("FAILURES:")
        for test, tb in result.failures + result.errors:
            print(f"  - {test}: {tb.splitlines()[-1]}")
    print(f"{'='*50}")

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
