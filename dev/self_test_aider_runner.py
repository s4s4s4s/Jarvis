"""P9: unit-тесты для aider_runner. Мокаем subprocess.run, проверяем контракт.

Эти тесты НЕ зовут реальный aider — только проверяют что наш wrapper:
  • правильно собирает argv
  • правильно интерпретирует exit codes
  • корректно обрабатывает timeout / FileNotFound
  • ретраит нужное число раз
  • читает результат с диска и возвращает его в content
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Bootstrap: подменить ollama чтобы можно было импортировать brain.*
import _test_bootstrap  # noqa: F401

from brain.agents import aider_runner
from brain.agents.aider_runner import AiderResult, aider_build, aider_heal, is_aider_available


class _TempProjectDir(unittest.TestCase):
    """Базовый класс с временной project_dir."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="aider_test_"))
        self.target = "main.py"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _fake_completed(self, rc: int = 0, stdout: str = "", stderr: str = ""):
        """Фабрика заглушки для subprocess.run."""
        m = MagicMock()
        m.returncode = rc
        m.stdout = stdout
        m.stderr = stderr
        return m

    def _write_target(self, content: str = "print('ok')\n"):
        (self.tmpdir / self.target).write_text(content, encoding="utf-8")


class TestAiderBuildHappyPath(_TempProjectDir):
    """rc=0 + файл создан → ok=True, content прочитан с диска."""

    def test_returns_ok_true_with_content(self):
        def fake_run(*args, **kwargs):
            # симулируем что aider создал файл
            self._write_target("print('hello jarvis')\n")
            return self._fake_completed(rc=0, stdout="files edited", stderr="")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "сделай hello jarvis", max_retries=0)

        self.assertTrue(result.ok)
        self.assertEqual(result.file_path, self.target)
        self.assertIn("hello jarvis", result.content)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.error, "")

    def test_argv_has_required_flags(self):
        captured = {}

        def fake_run(argv, *args, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs.get("env", {})
            self._write_target("x=1\n")
            return self._fake_completed(rc=0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            aider_build(self.tmpdir, self.target, "do stuff", max_retries=0)

        argv = captured["argv"]
        self.assertIn("--no-git", argv)
        self.assertIn("--yes-always", argv)
        self.assertIn("--message", argv)
        self.assertIn("--no-stream", argv)
        self.assertIn("--no-show-model-warnings", argv)
        self.assertIn(self.target, argv)
        # env: OLLAMA_API_BASE без /v1 (нативный эндпоинт), OPENAI_API_BASE с /v1
        env = captured["env"]
        self.assertIn("OLLAMA_API_BASE", env)
        self.assertFalse(env["OLLAMA_API_BASE"].endswith("/v1"),
                         "OLLAMA_API_BASE не должен иметь /v1 — это нативный эндпоинт")
        self.assertIn("OPENAI_API_BASE", env)
        self.assertTrue(env["OPENAI_API_BASE"].endswith("/v1"))
        self.assertIn("OPENAI_API_KEY", env)


class TestAiderBuildNonZeroExit(_TempProjectDir):
    """rc != 0 но файл не пустой → всё равно ok=True (aider иногда даёт warnings)."""

    def test_non_zero_rc_with_file_present_is_ok(self):
        def fake_run(*args, **kwargs):
            self._write_target("import sys\nprint(sys.argv)\n")
            return self._fake_completed(rc=2, stdout="", stderr="some warning")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "x", max_retries=0)

        self.assertTrue(result.ok)
        self.assertIn("import sys", result.content)


class TestAiderBuildEmptyFile(_TempProjectDir):
    """rc=0 но файл пустой → ok=False с понятным error."""

    def test_empty_file_after_clean_exit_is_failure(self):
        def fake_run(*args, **kwargs):
            (self.tmpdir / self.target).write_text("", encoding="utf-8")
            return self._fake_completed(rc=0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "x", max_retries=0)

        self.assertFalse(result.ok)
        self.assertIn("empty", result.error.lower())


class TestAiderBuildRetries(_TempProjectDir):
    """Первая попытка падает rc=1, вторая создаёт файл → ok=True, attempts=2."""

    def test_retries_on_failure_until_success(self):
        call_count = {"n": 0}

        def fake_run(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # первый раз ничего не пишем, rc=1
                return self._fake_completed(rc=1, stderr="parse error")
            # второй раз пишем файл
            self._write_target("ok\n")
            return self._fake_completed(rc=0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "x", max_retries=2)

        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(call_count["n"], 2)

    def test_gives_up_after_max_retries(self):
        call_count = {"n": 0}

        def fake_run(*args, **kwargs):
            call_count["n"] += 1
            return self._fake_completed(rc=1, stderr="always fails")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "x", max_retries=2)

        self.assertFalse(result.ok)
        # max_retries=2 → 3 попытки всего (1 + 2 ретрая)
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(result.attempts, 3)
        self.assertIn("after 3 attempts", result.error)


class TestAiderBuildTimeout(_TempProjectDir):
    """TimeoutExpired → exit_code=-1, ok=False, error содержит TIMEOUT."""

    def test_timeout_returns_failure_no_retry_loop_blocks(self):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "x", max_retries=0, timeout_s=1)

        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("TIMEOUT", result.stderr)


class TestAiderBuildBinaryMissing(_TempProjectDir):
    """FileNotFoundError → exit_code=-2, ok=False, без ретраев (aider не установлен)."""

    def test_missing_binary_no_retry(self):
        call_count = {"n": 0}

        def fake_run(*args, **kwargs):
            call_count["n"] += 1
            raise FileNotFoundError("aider not found")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_build(self.tmpdir, self.target, "x", max_retries=3)

        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, -2)
        # Один заход — ретраить отсутствующий бинарь бессмысленно
        self.assertEqual(call_count["n"], 1)


class TestAiderHeal(_TempProjectDir):
    """aider_heal — обёртка над build с error-инструкцией."""

    def test_heal_passes_error_text_to_aider(self):
        captured = {}

        def fake_run(argv, *args, **kwargs):
            # вытащим --message из argv
            idx = argv.index("--message")
            captured["msg"] = argv[idx + 1]
            self._write_target("print('fixed')\n")
            return self._fake_completed(rc=0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = aider_heal(self.tmpdir, self.target, "AssertionError: 1 != 2",
                                test_command="python main.py", max_retries=0)

        self.assertTrue(result.ok)
        self.assertIn("AssertionError", captured["msg"])
        self.assertIn("python main.py", captured["msg"])


class TestIsAiderAvailable(unittest.TestCase):
    """is_aider_available() корректно различает наличие/отсутствие бинаря."""

    def test_returns_true_when_aider_present(self):
        def fake_run(*args, **kwargs):
            m = MagicMock()
            m.returncode = 0
            return m

        with patch.object(subprocess, "run", side_effect=fake_run):
            self.assertTrue(is_aider_available())

    def test_returns_false_when_missing(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            self.assertFalse(is_aider_available())

    def test_returns_false_on_timeout(self):
        with patch.object(subprocess, "run",
                          side_effect=subprocess.TimeoutExpired(cmd="aider", timeout=1)):
            self.assertFalse(is_aider_available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
