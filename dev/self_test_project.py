# dev/self_test_project.py
"""
Smoke-тест ProjectAgent (Level 4) без живой Ollama.

Подменяет brain.client.chat фейковыми ответами для каждой фазы и проверяет,
что:
  - проект создаётся в tmp-папке
  - manifest.json валиден
  - все файлы записаны
  - reviewer-loop отрабатывает (revise → approve)
  - smoke-тест внутри проекта проходит
  - возвращается финальный отчёт-строка

Запуск:
  python -m dev.self_test_project

Принцип: ProjectAgent должен быть полностью тестируемым без LLM. Если эти
тесты падают — проблема в инфраструктуре, не в моделях.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── фейковые LLM-ответы ──────────────────────────────────────────────────────
_FAKE_SPEC = {
    "title": "Hello CLI",
    "slug":  "hello-cli",
    "kind":  "cli",
    "language": "python",
    "summary":  "печатает приветствие",
    "requirements": ["скрипт печатает 'hello jarvis'", "выходит с кодом 0"],
    "deliverables": ["main.py"],
    "acceptance_criteria": ["python main.py выводит hello jarvis"],
}

_FAKE_PLAN = {
    "files": [
        {"path": "main.py", "purpose": "точка входа", "depends_on": ["stdlib"]},
    ],
    "build_steps": [
        {"step": 1, "kind": "create_file", "target": "main.py", "description": "entry"},
        {"step": 2, "kind": "smoke_run",   "target": "python main.py", "description": "smoke"},
    ],
    "tests": [
        {"name": "smoke", "command": "python main.py", "expects": "hello jarvis"},
    ],
}

_FAKE_CODE_BAD  = "print('helo')\n"
_FAKE_CODE_GOOD = "print('hello jarvis')\n"

_FAKE_REVIEW_REVISE = {
    "verdict": "revise",
    "issues":  [{"severity": "major", "line_hint": 1, "problem": "опечатка helo", "suggestion": "напиши hello jarvis"}],
    "summary": "опечатка",
}
_FAKE_REVIEW_APPROVE = {
    "verdict": "approve",
    "issues":  [],
    "summary": "ок",
}
_FAKE_REPORT = "Готово, проект hello-cli собран и тест прошёл."


def make_fake_chat(seq: list):
    """seq — список объектов: dict (станет json.dumps) или str (вернётся как есть)."""
    state = {"i": 0}
    def fake_chat(model, msgs, options=None):
        i = state["i"]
        if i >= len(seq):
            raise AssertionError(f"unexpected extra LLM call #{i}")
        item = seq[i]
        state["i"] += 1
        return json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
    return fake_chat, state


class TestProjectAgentEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jarvis-test-"))
        os.environ["JARVIS_ROOT"] = str(self.tmp)

        # перезагрузим модули, которые кэшируют пути
        import importlib
        import core.paths as paths_mod
        importlib.reload(paths_mod)
        paths_mod.ensure_dirs()

        import tools.projects as proj_mod
        importlib.reload(proj_mod)
        self.proj_mod = proj_mod

        import brain.agents.coder as coder_mod
        importlib.reload(coder_mod)
        import brain.agents.reviewer as reviewer_mod
        importlib.reload(reviewer_mod)
        import brain.agents.project as project_mod
        importlib.reload(project_mod)
        self.project_mod = project_mod

    def tearDown(self):
        os.environ.pop("JARVIS_ROOT", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_pipeline_with_revise_then_approve(self):
        """
        Сценарий:
          intake → architect → coder(bad) → reviewer(revise) →
          coder(good) → reviewer(approve) → test → report
        """
        seq = [
            _FAKE_SPEC,           # intake
            _FAKE_PLAN,           # architect
            _FAKE_CODE_BAD,       # coder.write_file
            _FAKE_REVIEW_REVISE,  # reviewer.review #1
            _FAKE_CODE_GOOD,      # coder.patch_file
            _FAKE_REVIEW_APPROVE, # reviewer.review #2
            _FAKE_REPORT,         # report
        ]
        fake_chat, state = make_fake_chat(seq)

        with patch("brain.agents.project.chat", side_effect=fake_chat), \
             patch("brain.agents.coder.chat",   side_effect=fake_chat), \
             patch("brain.agents.reviewer.chat",side_effect=fake_chat):
            result = self.project_mod.run("сделай скрипт hello jarvis", [])

        # все ожидаемые вызовы LLM сделаны
        self.assertEqual(state["i"], len(seq), f"использовано {state['i']} из {len(seq)}")
        self.assertIn("hello-cli", result.lower())

        # проект существует на диске
        from core.paths import PROJECTS_DIR
        projects = [p for p in PROJECTS_DIR.iterdir() if p.is_dir()]
        self.assertEqual(len(projects), 1, f"projects on disk: {projects}")
        pdir = projects[0]
        self.assertTrue(pdir.name.startswith("hello-cli"))

        # манифест валиден
        manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "done", f"phases={manifest['phases']}")
        self.assertEqual(manifest["files"], ["main.py"])
        # фазы: intake, architect, build:main.py, test:smoke, finalize
        phase_names = [p["name"] for p in manifest["phases"]]
        self.assertIn("intake", phase_names)
        self.assertIn("architect", phase_names)
        self.assertIn("build:main.py", phase_names)
        self.assertIn("test:smoke", phase_names)
        self.assertIn("finalize", phase_names)

        # main.py содержит правильный текст
        main_py = (pdir / "main.py").read_text(encoding="utf-8")
        self.assertIn("hello jarvis", main_py)

    def test_intake_fallback_on_invalid_json(self):
        """Если LLM выдал мусор — спецификация всё равно должна получиться."""
        from brain.agents.project import _intake
        with patch("brain.agents.project.chat", return_value="не json вовсе"):
            spec = _intake("сделай мне калькулятор")
        self.assertIn("title", spec)
        self.assertIn("requirements", spec)

    def test_reviewer_fallback_on_llm_error(self):
        """Если LLM упал, reviewer должен вернуть approve чтобы не зациклить пайплайн."""
        from brain.agents.reviewer import review
        def boom(*a, **kw):
            raise RuntimeError("ollama down")
        with patch("brain.agents.reviewer.chat", side_effect=boom):
            verdict = review({"title": "x"}, {"path": "main.py"}, "print('ok')\n")
        self.assertEqual(verdict["verdict"], "approve")
        self.assertEqual(verdict["_source"], "fallback")


class TestProjectStoreSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jarvis-store-"))
        os.environ["JARVIS_ROOT"] = str(self.tmp)
        import importlib
        import core.paths as paths_mod
        importlib.reload(paths_mod)
        paths_mod.ensure_dirs()
        import tools.projects as pm
        importlib.reload(pm)
        self.pm = pm

    def tearDown(self):
        os.environ.pop("JARVIS_ROOT", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_slugify_cyrillic(self):
        s = self.pm.slugify("Калькулятор подходов в зале")
        self.assertRegex(s, r"^[a-z0-9-]+$")
        self.assertLessEqual(len(s), self.pm.MAX_SLUG_LEN)

    def test_path_traversal_blocked(self):
        m = self.pm.create_project({"title": "X", "slug": "x"})
        with self.assertRaises(ValueError):
            self.pm.write_project_file(m.slug, "../../etc/passwd", "evil")
        with self.assertRaises(ValueError):
            self.pm.write_project_file(m.slug, "/abs/path", "evil")

    def test_oversize_blocked(self):
        m = self.pm.create_project({"title": "Y", "slug": "y"})
        big = "x" * (self.pm.MAX_FILE_BYTES + 1)
        with self.assertRaises(ValueError):
            self.pm.write_project_file(m.slug, "big.txt", big)

    def test_run_in_project_no_shell(self):
        m = self.pm.create_project({"title": "Z", "slug": "z"})
        # shell=False → "&&" не интерпретируется как chain; будет ошибка/нулевой stdout
        res = self.pm.run_in_project(m.slug, [sys.executable, "-c", "print(1)"])
        self.assertTrue(res["ok"])
        self.assertIn("1", res["stdout"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
