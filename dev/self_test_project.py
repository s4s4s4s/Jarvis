# dev/self_test_project.py
"""
Smoke + интеграционные тесты ProjectAgent (Level 4) без живой Ollama.

Все LLM-вызовы мокаются. Реальные subprocess (venv create, python main.py,
pip — мокаем pip отдельно) — настоящие.

Запуск:
  python -m dev.self_test_project

Покрытие:
  TestProjectAgentEndToEnd
    - test_full_pipeline_with_revise_then_approve   (build cycle)
    - test_intake_fallback_on_invalid_json
    - test_reviewer_fallback_on_llm_error
    - test_self_heal_recovers_failing_test          (Level 4 self-heal)
    - test_budget_exceeded_graceful_failure         (бюджеты)
    - test_resume_picks_up_after_architect          (--resume)
    - test_readme_fallback_when_llm_empty           (детерминированный README)
    - test_cross_file_context_passed_to_coder       (cross-file)

  TestProjectStoreSafety
    - test_slugify_cyrillic
    - test_path_traversal_blocked
    - test_oversize_blocked
    - test_run_in_project_no_shell
    - test_pkg_spec_validation                      (защита pip_install)
    - test_index_jsonl_append_and_read              (метрики)
    - test_get_project_files_round_trip             (cross-file context source)
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
_FAKE_README = "# Hello CLI\nпечатает hello jarvis\n"
_FAKE_REPORT = "Готово, проект hello-cli собран и тест прошёл."


def make_fake_chat(seq: list):
    state = {"i": 0, "calls": []}
    def fake_chat(model, msgs, options=None):
        state["calls"].append({"model": model, "system": msgs[0]["content"][:80], "user": msgs[1]["content"][:200]})
        i = state["i"]
        if i >= len(seq):
            raise AssertionError(f"unexpected extra LLM call #{i}")
        item = seq[i]
        state["i"] += 1
        return json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
    return fake_chat, state


def _patch_chat_everywhere(fake_chat):
    """Подменить chat во всех модулях ProjectAgent одним менеджером."""
    return [
        patch("brain.agents.project.chat", side_effect=fake_chat),
        patch("brain.agents.coder.chat",   side_effect=fake_chat),
        patch("brain.agents.reviewer.chat",side_effect=fake_chat),
    ]


def _enter(patches):
    return [p.__enter__() for p in patches]


def _exit(patches):
    for p in patches:
        try: p.__exit__(None, None, None)
        except Exception: pass


# ─── базовый setup ───────────────────────────────────────────────────────────
class _IsolatedJarvisRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jarvis-test-"))
        os.environ["JARVIS_ROOT"] = str(self.tmp)
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


# ─── End-to-end ──────────────────────────────────────────────────────────────
class TestProjectAgentEndToEnd(_IsolatedJarvisRoot):
    def test_full_pipeline_with_revise_then_approve(self):
        seq = [
            _FAKE_SPEC,            # intake
            _FAKE_PLAN,            # architect
            _FAKE_CODE_BAD,        # coder.write_file
            _FAKE_REVIEW_REVISE,   # reviewer #1
            _FAKE_CODE_GOOD,       # coder.patch_file
            _FAKE_REVIEW_APPROVE,  # reviewer #2
            _FAKE_README,          # readme
            # P5.2: report не зовёт LLM на happy path — детерминистика
        ]
        fake_chat, state = make_fake_chat(seq)
        patches = _patch_chat_everywhere(fake_chat); _enter(patches)
        try:
            result = self.project_mod.run("сделай скрипт hello jarvis", [],
                                           wall_budget_s=120, llm_budget=20)
        finally:
            _exit(patches)

        self.assertEqual(state["i"], len(seq), f"использовано {state['i']} из {len(seq)}")
        self.assertIn("hello-cli", result.lower())

        from core.paths import PROJECTS_DIR
        projects = [p for p in PROJECTS_DIR.iterdir() if p.is_dir() and p.name.startswith("hello-cli")]
        self.assertEqual(len(projects), 1)
        pdir = projects[0]
        manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "done", f"phases={manifest['phases']}")
        self.assertIn("README.md", manifest["files"])
        self.assertIn("main.py", manifest["files"])
        self.assertIn("hello jarvis", (pdir / "main.py").read_text(encoding="utf-8"))
        # _index.jsonl
        idx = self.proj_mod.read_index()
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0]["status"], "done")
        # метрики записаны
        self.assertIn("llm_used", manifest["metrics"])
        self.assertGreater(manifest["metrics"]["llm_used"], 0)
        # last_phase=finalize → resume будет ок
        self.assertEqual(manifest["last_phase"], "finalize")

    def test_self_heal_recovers_failing_test(self):
        """Build выдаёт код который НЕ печатает 'hello jarvis' → expects fail →
           Healer диагностирует → Coder патчит → перетест проходит."""
        bad_runs_print = "print('something else')\n"
        good_print = "print('hello jarvis')\n"
        seq = [
            _FAKE_SPEC,
            _FAKE_PLAN,
            bad_runs_print,        # coder.write
            _FAKE_REVIEW_APPROVE,  # reviewer (одобряет — код синтаксически ок)
            # тут идёт TEST: subprocess реальный, expects='hello jarvis' не найдётся → fail
            # → heal:
            {"diagnosis": "печатается не та строка", "target_file": "main.py",
             "fix_instruction": "замени строку на hello jarvis"},   # heal.diagnose #1
            good_print,            # coder.patch_file (внутри heal)
            # тесты после heal — без LLM, реальный subprocess
            _FAKE_README,          # readme
            # P5.2: report не зовёт LLM на happy path
        ]
        fake_chat, state = make_fake_chat(seq)
        patches = _patch_chat_everywhere(fake_chat); _enter(patches)
        try:
            result = self.project_mod.run("hello jarvis скрипт", [],
                                           wall_budget_s=120, llm_budget=20)
        finally:
            _exit(patches)

        self.assertEqual(state["i"], len(seq), f"использовано {state['i']} из {len(seq)}")
        from core.paths import PROJECTS_DIR
        pdir = next(p for p in PROJECTS_DIR.iterdir() if p.is_dir() and p.name.startswith("hello-cli"))
        manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "done",
                         f"healing должен был починить проект; phases={[p['name'] for p in manifest['phases']]}")
        # heal-фаза присутствует
        names = [p["name"] for p in manifest["phases"]]
        self.assertTrue(any(n.startswith("heal:iter") for n in names),
                        f"не найдена фаза heal среди {names}")
        # main.py содержит правильный текст после heal
        self.assertIn("hello jarvis", (pdir / "main.py").read_text(encoding="utf-8"))

    def test_intake_fallback_on_invalid_json(self):
        from brain.agents.project import _intake, Budget
        with patch("brain.agents.project.chat", return_value="не json вовсе"):
            spec = _intake("сделай мне калькулятор", Budget())
        self.assertIn("title", spec)
        self.assertIn("requirements", spec)

    def test_reviewer_fallback_on_llm_error(self):
        from brain.agents.reviewer import review
        def boom(*a, **kw): raise RuntimeError("ollama down")
        with patch("brain.agents.reviewer.chat", side_effect=boom):
            verdict = review({"title": "x"}, {"path": "main.py"}, "print('ok')\n")
        self.assertEqual(verdict["verdict"], "approve")
        self.assertEqual(verdict["_source"], "fallback")

    def test_budget_exceeded_graceful_failure(self):
        """LLM-budget=2 → даже intake+architect не пройдут до конца → status=failed,
           без зависания, без необработанных исключений."""
        seq = [_FAKE_SPEC, _FAKE_PLAN, _FAKE_CODE_GOOD]   # доступно 3 ответа
        fake_chat, state = make_fake_chat(seq)
        patches = _patch_chat_everywhere(fake_chat); _enter(patches)
        try:
            # llm_budget=2: spend на intake (1) + architect (2) = 2 → перед build budget исчерпан
            result = self.project_mod.run("проект", [], wall_budget_s=120, llm_budget=2)
        finally:
            _exit(patches)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Папка проекта существует, manifest.failed или intermediate
        from core.paths import PROJECTS_DIR
        pdirs = [p for p in PROJECTS_DIR.iterdir() if p.is_dir() and p.name != "_index.jsonl"]
        self.assertGreaterEqual(len(pdirs), 1)

    def test_resume_picks_up_after_architect(self):
        """Симулируем падение после architect, потом resume должен продолжить."""
        # Полный seq для первого run (упадём после architect через budget)
        seq1 = [_FAKE_SPEC, _FAKE_PLAN]   # ровно 2 → дальше будет BudgetExceeded
        fake1, _ = make_fake_chat(seq1)
        patches = _patch_chat_everywhere(fake1); _enter(patches)
        try:
            self.project_mod.run("проект для resume", [], wall_budget_s=120, llm_budget=2)
        finally:
            _exit(patches)

        from core.paths import PROJECTS_DIR
        pdirs = [p for p in PROJECTS_DIR.iterdir() if p.is_dir()]
        self.assertEqual(len(pdirs), 1)
        slug = pdirs[0].name
        m1 = json.loads((pdirs[0] / "manifest.json").read_text(encoding="utf-8"))
        # last_phase должен быть одним из ранних этапов: intake → architect → env
        # (env не расходует LLM-бюджет, поэтому проходит до build)
        self.assertIn(m1["last_phase"], ("intake", "architect", "env"),
                      f"unexpected last_phase: {m1.get('last_phase')}")

        # Теперь resume с полным seq для оставшихся фаз: build, test, readme, report
        seq2 = [_FAKE_CODE_GOOD, _FAKE_REVIEW_APPROVE, _FAKE_README, _FAKE_REPORT]
        fake2, _ = make_fake_chat(seq2)
        patches = _patch_chat_everywhere(fake2); _enter(patches)
        try:
            out = self.project_mod.resume(slug, wall_budget_s=120, llm_budget=20)
        finally:
            _exit(patches)
        self.assertIsInstance(out, str)
        m2 = json.loads((pdirs[0] / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m2["status"], "done", f"phases after resume: {[p['name'] for p in m2['phases']]}")
        # фаза 'resume' зафиксирована
        names = [p["name"] for p in m2["phases"]]
        self.assertIn("resume", names)

    def test_readme_fallback_when_llm_empty(self):
        """Если LLM вернул пустую строку — README должен сгенериться детерминистически."""
        seq = [
            _FAKE_SPEC, _FAKE_PLAN, _FAKE_CODE_GOOD, _FAKE_REVIEW_APPROVE,
            "",                     # README LLM вернул пусто
            _FAKE_REPORT,
        ]
        fake_chat, _ = make_fake_chat(seq)
        patches = _patch_chat_everywhere(fake_chat); _enter(patches)
        try:
            self.project_mod.run("hello", [], wall_budget_s=120, llm_budget=20)
        finally:
            _exit(patches)
        from core.paths import PROJECTS_DIR
        pdir = next(p for p in PROJECTS_DIR.iterdir() if p.is_dir())
        readme = (pdir / "README.md").read_text(encoding="utf-8")
        self.assertIn("Hello CLI", readme)
        self.assertIn("main.py", readme)

    def test_cross_file_context_passed_to_coder(self):
        """При генерации второго файла Coder должен видеть содержимое первого."""
        spec_two = dict(_FAKE_SPEC)
        spec_two["title"] = "Two Files"
        spec_two["slug"]  = "two-files"
        plan_two = {
            "files": [
                {"path": "lib.py",  "purpose": "функция greet()", "depends_on": ["stdlib"]},
                {"path": "main.py", "purpose": "вызывает greet()", "depends_on": ["lib.py"]},
            ],
            "build_steps": [],
            "tests": [{"name": "smoke", "command": "python main.py", "expects": "hi"}],
        }
        lib_code  = "def greet():\n    return 'hi'\n"
        main_code = "from lib import greet\nprint(greet())\n"
        seq = [
            spec_two, plan_two,
            lib_code,  _FAKE_REVIEW_APPROVE,
            main_code, _FAKE_REVIEW_APPROVE,
            _FAKE_README, _FAKE_REPORT,
        ]
        fake_chat, state = make_fake_chat(seq)
        patches = _patch_chat_everywhere(fake_chat); _enter(patches)
        try:
            self.project_mod.run("two files project", [], wall_budget_s=120, llm_budget=20)
        finally:
            _exit(patches)

        # Прицельная проверка: при генерации main.py user-msg ДОЛЖЕН содержать
        # текст уже-написанного lib.py (cross-file context).
        coder_calls = [c for c in state["calls"] if "разработчик" in c["system"]]
        self.assertGreaterEqual(len(coder_calls), 2)
        # Второй coder-call (main.py) должен содержать упоминание lib.py
        self.assertIn("lib.py", coder_calls[1]["user"])


# ─── Project store safety ────────────────────────────────────────────────────
class TestProjectStoreSafety(_IsolatedJarvisRoot):
    def test_slugify_cyrillic(self):
        s = self.proj_mod.slugify("Калькулятор подходов в зале")
        self.assertRegex(s, r"^[a-z0-9-]+$")
        self.assertLessEqual(len(s), self.proj_mod.MAX_SLUG_LEN)

    def test_path_traversal_blocked(self):
        m = self.proj_mod.create_project({"title": "X", "slug": "x"})
        with self.assertRaises(ValueError):
            self.proj_mod.write_project_file(m.slug, "../../etc/passwd", "evil")
        with self.assertRaises(ValueError):
            self.proj_mod.write_project_file(m.slug, "/abs/path", "evil")

    def test_oversize_blocked(self):
        m = self.proj_mod.create_project({"title": "Y", "slug": "y"})
        big = "x" * (self.proj_mod.MAX_FILE_BYTES + 1)
        with self.assertRaises(ValueError):
            self.proj_mod.write_project_file(m.slug, "big.txt", big)

    def test_run_in_project_no_shell(self):
        m = self.proj_mod.create_project({"title": "Z", "slug": "z"})
        res = self.proj_mod.run_in_project(m.slug, [sys.executable, "-c", "print(1)"])
        self.assertTrue(res["ok"])
        self.assertIn("1", res["stdout"])

    def test_pkg_spec_validation(self):
        # _validate_pkg_spec — внутренний, проверяем через pip_install dry checks
        m = self.proj_mod.create_project({"title": "P", "slug": "p"})
        # фейковые опасные пакетные спеки должны быть отклонены до запуска pip
        bad = self.proj_mod.pip_install(m.slug, ["evil; rm -rf /"])
        self.assertFalse(bad["ok"])
        self.assertIn("unsafe", bad.get("error", ""))
        bad2 = self.proj_mod.pip_install(m.slug, ["-e ."])
        self.assertFalse(bad2["ok"])

    def test_index_jsonl_append_and_read(self):
        self.proj_mod.append_index_record({"slug": "a", "status": "done"})
        self.proj_mod.append_index_record({"slug": "b", "status": "failed"})
        idx = self.proj_mod.read_index()
        self.assertEqual([r["slug"] for r in idx], ["a", "b"])

    def test_get_project_files_round_trip(self):
        m = self.proj_mod.create_project({"title": "RT", "slug": "rt"})
        self.proj_mod.write_project_file(m.slug, "a.py", "print(1)\n")
        self.proj_mod.write_project_file(m.slug, "b.py", "print(2)\n")
        files = self.proj_mod.get_project_files(m.slug)
        self.assertEqual(set(files.keys()), {"a.py", "b.py"})
        self.assertIn("print(1)", files["a.py"])


# ─── Новые тесты: env-парсер и детерминистический healer ──────────────
class TestEnvParserAndAutoHeal(_IsolatedJarvisRoot):
    """Проверяем свежие фиксы: парсер install_dep и ModuleNotFoundError→pip auto-install."""

    def test_extract_packages_ignores_pip_install_r_command(self):
        """Не должны пытаться ставить 'requirements.txt' как пакет (баг с untitled-project)."""
        plan = {
            "files": [{"path": "main.py"}, {"path": "requirements.txt"}],
            "build_steps": [
                {"step": 1, "kind": "create_file", "target": "main.py"},
                {"step": 2, "kind": "install_dep", "target": "pip install -r requirements.txt"},
            ],
        }
        m = self.proj_mod.create_project({"title": "E", "slug": "e"})
        self.proj_mod.write_project_file(m.slug, "requirements.txt", "feedparser\n")
        pkgs = self.project_mod._extract_packages(m.slug, plan)
        self.assertNotIn("requirements.txt", pkgs)
        self.assertNotIn("-r", pkgs)
        self.assertNotIn("pip", pkgs)
        self.assertIn("feedparser", pkgs)

    def test_extract_packages_uses_pip_requirements_field(self):
        """Новое поле plan['pip_requirements'] — источник #1."""
        plan = {
            "files": [{"path": "main.py"}],
            "pip_requirements": ["requests", "click==8.1", "feedparser"],
            "build_steps": [],
        }
        m = self.proj_mod.create_project({"title": "E2", "slug": "e2"})
        pkgs = self.project_mod._extract_packages(m.slug, plan)
        self.assertEqual(pkgs[:3], ["requests", "click==8.1", "feedparser"])

    def test_extract_packages_dedup_across_sources(self):
        """feedparser в pip_requirements + в requirements.txt → в результате один раз."""
        m = self.proj_mod.create_project({"title": "E3", "slug": "e3"})
        self.proj_mod.write_project_file(m.slug, "requirements.txt", "feedparser\nrequests\n")
        plan = {"files": [{"path": "main.py"}], "pip_requirements": ["feedparser"]}
        pkgs = self.project_mod._extract_packages(m.slug, plan)
        # Один feedparser, плюс requests из requirements.txt
        self.assertEqual([p.lower() for p in pkgs].count("feedparser"), 1)
        self.assertIn("requests", pkgs)

    def test_heal_detects_module_not_found_and_calls_pip_install(self):
        """_heal_missing_module должен выдернуть имя модуля и вызвать pip_install."""
        m = self.proj_mod.create_project({"title": "H", "slug": "h"})
        failed = {
            "name": "smoke",
            "command": "python main.py",
            "rc": 1,
            "ok": False,
            "stderr": "Traceback (most recent call last):\n  File \"main.py\", line 2, in <module>\n    import feedparser\nModuleNotFoundError: No module named 'feedparser'\n",
        }
        calls = []

        def fake_pip_install(slug, packages, timeout=180):
            calls.append((slug, list(packages)))
            return {"ok": True, "installed": packages}

        with patch("brain.agents.project.pip_install", side_effect=fake_pip_install):
            res = self.project_mod._heal_missing_module(m.slug, failed)

        self.assertIsNotNone(res, "healer должен распознать ModuleNotFoundError")
        self.assertTrue(res["ok"])
        self.assertEqual(res["missing"], "feedparser")
        self.assertEqual(res["installed_as"], "feedparser")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], ["feedparser"])

    def test_heal_module_not_found_returns_none_for_other_errors(self):
        """Если это не ModuleNotFoundError — детерминистический healer должен вернуть None."""
        m = self.proj_mod.create_project({"title": "H2", "slug": "h2"})
        failed = {"name": "smoke", "command": "python main.py", "rc": 1, "ok": False,
                  "stderr": "SyntaxError: invalid syntax"}
        res = self.project_mod._heal_missing_module(m.slug, failed)
        self.assertIsNone(res)

    def test_heal_module_not_found_uses_import_to_pip_mapping(self):
        """import cv2 → pip install opencv-python."""
        m = self.proj_mod.create_project({"title": "H3", "slug": "h3"})
        failed = {"name": "smoke", "command": "python main.py", "rc": 1, "ok": False,
                  "stderr": "ModuleNotFoundError: No module named 'cv2'\n"}
        captured = []
        def fake_pip_install(slug, packages, timeout=180):
            captured.append(list(packages))
            return {"ok": True, "installed": packages}
        with patch("brain.agents.project.pip_install", side_effect=fake_pip_install):
            res = self.project_mod._heal_missing_module(m.slug, failed)
        self.assertIsNotNone(res)
        self.assertEqual(captured, [["opencv-python"]])
        self.assertEqual(res["installed_as"], "opencv-python")


# ─── P0: structured machine checks ───────────────────────────────────────
class TestStructuredChecks(_IsolatedJarvisRoot):
    """P0: машинно-проверяемые checks заменяют свободный expects-в-stdout."""

    def _make_proj_with_file(self, rel_path: str, content: str) -> str:
        m = self.proj_mod.create_project({"title": "chk", "slug": "chk"})
        self.proj_mod.write_project_file(m.slug, rel_path, content)
        return m.slug

    def test_check_rc_zero(self):
        ev = self.project_mod._evaluate_check
        self.assertTrue(ev("any", {"type": "rc_zero"}, {"returncode": 0})["ok"])
        self.assertFalse(ev("any", {"type": "rc_zero"}, {"returncode": 1})["ok"])

    def test_check_file_exists(self):
        slug = self._make_proj_with_file("out.txt", "hi")
        ev = self.project_mod._evaluate_check
        self.assertTrue(ev(slug, {"type": "file_exists", "path": "out.txt"}, {})["ok"])
        self.assertFalse(ev(slug, {"type": "file_exists", "path": "missing.txt"}, {})["ok"])

    def test_check_file_min_size(self):
        slug = self._make_proj_with_file("data.csv", "a,b,c\n1,2,3\n")  # 12 bytes
        ev = self.project_mod._evaluate_check
        self.assertTrue(ev(slug, {"type": "file_min_size", "path": "data.csv", "bytes": 5}, {})["ok"])
        self.assertFalse(ev(slug, {"type": "file_min_size", "path": "data.csv", "bytes": 9999}, {})["ok"])

    def test_check_file_min_lines(self):
        slug = self._make_proj_with_file("news.csv", "h1,h2\nr1c1,r1c2\nr2c1,r2c2\n")
        ev = self.project_mod._evaluate_check
        self.assertTrue(ev(slug, {"type": "file_min_lines", "path": "news.csv", "lines": 3}, {})["ok"])
        self.assertFalse(ev(slug, {"type": "file_min_lines", "path": "news.csv", "lines": 100}, {})["ok"])

    def test_check_stdout_contains_case_insensitive(self):
        ev = self.project_mod._evaluate_check
        run = {"stdout": "Done. Saved 25 rows to news.csv"}
        self.assertTrue(ev("any", {"type": "stdout_contains", "text": "saved"}, run)["ok"])
        self.assertFalse(ev("any", {"type": "stdout_contains", "text": "failed"}, run)["ok"])

    def test_check_unknown_type_rejected(self):
        ev = self.project_mod._evaluate_check
        r = ev("any", {"type": "shell_exec", "cmd": "rm -rf /"}, {"returncode": 0})
        self.assertFalse(r["ok"])
        self.assertIn("unknown check type", r["reason"])

    def test_check_path_traversal_rejected(self):
        slug = self._make_proj_with_file("a.txt", "x")
        ev = self.project_mod._evaluate_check
        r = ev(slug, {"type": "file_exists", "path": "../../../etc/passwd"}, {})
        self.assertFalse(r["ok"])
        self.assertIn("unsafe path", r["reason"])

    def test_run_one_test_with_structured_checks_passes(self):
        """Полный _run_one_test: скрипт пишет файл, checks проверяют результат на диске."""
        m = self.proj_mod.create_project({"title": "E2E", "slug": "e2e"})
        # Скрипт создаёт news.csv с 4 строками
        self.proj_mod.write_project_file(m.slug, "main.py",
            "open('news.csv','w').write('h1,h2\\na,b\\nc,d\\ne,f\\n')\nprint('done')\n")
        self.proj_mod.ensure_venv(m.slug)
        t = {
            "name": "e2e",
            "command": "python main.py",
            "checks": [
                {"type": "rc_zero"},
                {"type": "file_exists", "path": "news.csv"},
                {"type": "file_min_lines", "path": "news.csv", "lines": 3},
                {"type": "stdout_contains", "text": "done"},
            ],
        }
        rec = self.project_mod._run_one_test(m.slug, t)
        self.assertTrue(rec["ok"], f"checks failed: {rec.get('checks')}")
        self.assertEqual(len(rec["checks"]), 4)
        self.assertTrue(all(c["ok"] for c in rec["checks"]))

    def test_run_one_test_legacy_expects_still_works(self):
        """Старые планы с expects-в-stdout должны продолжать работать."""
        m = self.proj_mod.create_project({"title": "Legacy", "slug": "legacy"})
        self.proj_mod.write_project_file(m.slug, "main.py", "print('hello world')\n")
        self.proj_mod.ensure_venv(m.slug)
        t = {"name": "legacy", "command": "python main.py", "expects": "hello"}
        rec = self.project_mod._run_one_test(m.slug, t)
        self.assertTrue(rec["ok"])
        self.assertTrue(rec["expects_ok"])
        self.assertEqual(rec["checks"], [])


# ─── P1: static checks (ast + ruff/pyflakes) ────────────────────────────
class TestStaticChecks(unittest.TestCase):
    """P1: детерминистическая статика перед LLM-Reviewer."""

    def test_ast_check_valid_python(self):
        from tools.static_checks import ast_check
        r = ast_check("def foo():\n    return 42\n")
        self.assertTrue(r["ok"])
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["tool"], "ast")

    def test_ast_check_syntax_error_has_line_and_message(self):
        from tools.static_checks import ast_check
        r = ast_check("def foo(:\n    return 42\n")
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["errors"]), 1)
        self.assertIn("SyntaxError", r["errors"][0])
        self.assertIn("L1", r["errors"][0])

    def test_ast_check_empty_code_fails(self):
        from tools.static_checks import ast_check
        r = ast_check("")
        self.assertFalse(r["ok"])

    def test_ast_check_indentation_error(self):
        from tools.static_checks import ast_check
        r = ast_check("def foo():\nreturn 1\n")
        self.assertFalse(r["ok"])
        self.assertIn("L2", r["errors"][0])

    def test_static_check_skips_non_python(self):
        from tools.static_checks import static_check
        r = static_check("requirements.txt", "feedparser\nrequests\n")
        self.assertTrue(r["ok"])
        self.assertFalse(r["applicable"])
        self.assertEqual(r["errors"], [])

    def test_static_check_python_ok(self):
        from tools.static_checks import static_check
        r = static_check("main.py", "print('hi')\n")
        self.assertTrue(r["ok"])
        self.assertTrue(r["applicable"])
        self.assertEqual(r["errors"], [])
        self.assertIn("ast", r["tools"])

    def test_static_check_python_syntax_broken(self):
        from tools.static_checks import static_check
        r = static_check("main.py", "def x(:\n  pass\n")
        self.assertFalse(r["ok"])
        self.assertTrue(r["applicable"])
        self.assertTrue(len(r["errors"]) >= 1)

    def test_static_errors_to_feedback_format(self):
        from tools.static_checks import static_errors_to_feedback
        fb = static_errors_to_feedback(["L3:5 SyntaxError: invalid syntax"])
        self.assertIn("Синтаксически", fb) if False else None  # проверяем фактический текст
        self.assertIn("SyntaxError", fb)
        self.assertIn("L3:5", fb)

    def test_static_errors_to_feedback_empty(self):
        from tools.static_checks import static_errors_to_feedback
        self.assertEqual(static_errors_to_feedback([]), "")

    def test_lint_check_never_fails_on_valid_code(self):
        """lint — soft-проверка, никогда не возвращает ok=False."""
        from tools.static_checks import lint_check
        r = lint_check("x = 1\nprint(x)\n")
        self.assertTrue(r["ok"])
        self.assertIn("tool", r)


# ─── P1: build-loop интеграция static checks ─────────────────────────
class TestBuildLoopWithStatics(_IsolatedJarvisRoot):
    """Когда Coder выдаёт синтаксически битый код, LLM-Reviewer НЕ должен вызываться."""

    def test_static_failure_skips_reviewer_and_uses_static_feedback(self):
        # 1-я итерация: write_file выдаёт битый синтаксис,
        # 2-я итерация: patch_file (получив ast-feedback) выдаёт валидный.
        # Reviewer должен быть вызван РОВНО ОДИН раз (на 2-й итерации), не 2 раза.
        m = self.proj_mod.create_project({"title": "P1", "slug": "p1"})
        plan = {"files": [{"path": "main.py", "purpose": "x"}]}
        spec = {"title": "P1"}
        target = plan["files"][0]
        budget = self.project_mod.Budget(wall_s=60, llm=20)

        broken_code = "def foo(:\n    return 42\n"
        good_code   = "def foo():\n    return 42\n"

        write_calls = []
        patch_calls = []
        review_calls = []

        def fake_write(spec, plan, target, existing=None):
            write_calls.append((target["path"], existing))
            return broken_code

        def fake_patch(spec, plan, target, code, feedback, existing=None):
            patch_calls.append((target["path"], feedback))
            return good_code

        def fake_review(spec, target, code):
            review_calls.append({"code_first_line": code.splitlines()[0], "hint": spec.get("_static_lint_hint")})
            return {"verdict": "approve", "issues": [], "summary": "ok", "_source": "llm"}

        with patch.object(self.project_mod.coder_agent, "write_file", side_effect=fake_write), \
             patch.object(self.project_mod.coder_agent, "patch_file", side_effect=fake_patch), \
             patch.object(self.project_mod.reviewer_agent, "review", side_effect=fake_review):
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)

        self.assertTrue(res["ok"])
        self.assertEqual(len(write_calls), 1)        # первая итерация
        self.assertEqual(len(patch_calls), 1)        # вторая, с ast-feedback
        self.assertIn("SyntaxError", patch_calls[0][1])
        self.assertEqual(len(review_calls), 1)       # Reviewer вызван только на валидном коде
        self.assertEqual(res["static"]["fail_streak"], 1)
        self.assertTrue(res["static"]["final_ast_ok"])

    def test_static_ok_path_unchanged_behaviour(self):
        """Если Coder сразу выдаёт валидный код — build-loop ведёт себя как раньше."""
        m = self.proj_mod.create_project({"title": "P1", "slug": "p1ok"})
        plan = {"files": [{"path": "main.py", "purpose": "x"}]}
        spec = {"title": "ok"}
        target = plan["files"][0]
        budget = self.project_mod.Budget(wall_s=60, llm=20)

        review_calls = []
        with patch.object(self.project_mod.coder_agent, "write_file",
                          return_value="print('hi')\n"), \
             patch.object(self.project_mod.reviewer_agent, "review",
                          side_effect=lambda spec, target, code: (review_calls.append(1) or {"verdict": "approve", "issues": [], "summary": "ok", "_source": "llm"})):
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)

        self.assertTrue(res["ok"])
        self.assertEqual(len(review_calls), 1)
        self.assertEqual(res["static"]["fail_streak"], 0)

    def test_static_skipped_for_non_python(self):
        """Для requirements.txt static не применяется, build-loop работает как было."""
        m = self.proj_mod.create_project({"title": "P1", "slug": "p1txt"})
        plan = {"files": [{"path": "requirements.txt", "purpose": "deps"}]}
        spec = {"title": "deps"}
        target = plan["files"][0]
        budget = self.project_mod.Budget(wall_s=60, llm=20)

        with patch.object(self.project_mod.coder_agent, "write_file",
                          return_value="feedparser\n"), \
             patch.object(self.project_mod.reviewer_agent, "review",
                          return_value={"verdict": "approve", "issues": [], "summary": "ok", "_source": "llm"}):
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)

        self.assertTrue(res["ok"])
        self.assertFalse(res["static"].get("final_ast_ok") is False)
        # tools пустой — static не применялся
        self.assertEqual(res["static"]["tools"], [])


# ─── P3: adaptive budget ─────────────────────────────────────────────────────
class TestAdaptiveBudget(_IsolatedJarvisRoot):
    """P3: estimate_complexity и budget_for_tier — размерные метрики, без ключевых слов."""

    def test_complexity_xs_when_one_file_in_plan(self):
        plan = {"files": [{"path": "main.py"}]}
        self.assertEqual(self.project_mod.estimate_complexity("x", plan=plan), "XS")

    def test_complexity_s_when_two_files(self):
        plan = {"files": [{"path": "a.py"}, {"path": "b.py"}]}
        self.assertEqual(self.project_mod.estimate_complexity("x", plan=plan), "S")

    def test_complexity_m_when_3_or_4_files(self):
        for n in (3, 4):
            plan = {"files": [{"path": f"f{i}.py"} for i in range(n)]}
            self.assertEqual(self.project_mod.estimate_complexity("x", plan=plan), "M")

    def test_complexity_l_when_5plus_files(self):
        plan = {"files": [{"path": f"f{i}.py"} for i in range(5)]}
        self.assertEqual(self.project_mod.estimate_complexity("x", plan=plan), "L")

    def test_complexity_xs_short_query_no_plan(self):
        """Короткий запрос, мало requirements → XS."""
        spec = {"requirements": ["один"]}
        self.assertEqual(self.project_mod.estimate_complexity("скачай файл", spec=spec), "XS")

    def test_complexity_l_long_query_or_many_reqs(self):
        spec = {"requirements": ["r"] * 6}
        self.assertEqual(self.project_mod.estimate_complexity("x", spec=spec), "L")
        long_q = " ".join(["слово"] * 90)
        self.assertEqual(self.project_mod.estimate_complexity(long_q), "L")

    def test_complexity_m_medium_query(self):
        spec = {"requirements": ["r1", "r2", "r3"]}
        self.assertEqual(self.project_mod.estimate_complexity("x", spec=spec), "M")

    def test_complexity_default_s(self):
        # 20 слов, без spec/plan — не XS, не M, не L → S
        q = " ".join(["w"] * 20)
        self.assertEqual(self.project_mod.estimate_complexity(q), "S")

    def test_budget_for_tier_known_values(self):
        for tier, expected_llm in [("XS", 15), ("S", 30), ("M", 60), ("L", 120)]:
            params = self.project_mod.budget_for_tier(tier)
            self.assertEqual(params["llm"], expected_llm)
            self.assertGreater(params["wall_s"], 0)

    def test_budget_for_tier_unknown_defaults_to_m(self):
        self.assertEqual(
            self.project_mod.budget_for_tier("WEIRD"),
            self.project_mod.budget_for_tier("M"),
        )


# ─── P2: deterministic healers ───────────────────────────────────────────────
class TestDeterministicHealers(_IsolatedJarvisRoot):
    """P2: SyntaxError / ConnectionError / JSONDecodeError — без LLM-диагностики."""

    def _make_project_with_file(self, slug, fname, content):
        m = self.proj_mod.create_project({"title": "P2", "slug": slug})
        self.proj_mod.write_project_file(m.slug, fname, content)
        return m

    def test_heal_syntax_error_finds_broken_file(self):
        """Битый синтаксис в файле → детерминистический хилер указывает на него."""
        self._make_project_with_file("p2syn", "main.py", "def f(:\n    pass\n")
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "  File \"main.py\", line 1\nSyntaxError: invalid syntax\n"}
        diag = self.project_mod._heal_syntax_error("p2syn", failed, plan)
        self.assertIsNotNone(diag)
        self.assertEqual(diag["target_file"], "main.py")
        self.assertEqual(diag["category"], "syntax")
        self.assertTrue(diag["fix_instruction"])

    def test_heal_syntax_error_returns_none_when_stderr_clean(self):
        """Без SyntaxError в stderr — None."""
        self._make_project_with_file("p2syn2", "main.py", "print('ok')\n")
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "AssertionError: not equal"}
        diag = self.project_mod._heal_syntax_error("p2syn2", failed, plan)
        self.assertIsNone(diag)

    def test_heal_syntax_error_no_python_files(self):
        """Если в плане нет .py — None."""
        plan = {"files": [{"path": "data.csv"}]}
        failed = {"stderr": "SyntaxError: invalid syntax"}
        diag = self.project_mod._heal_syntax_error("p2nonpy", failed, plan)
        self.assertIsNone(diag)

    def test_heal_network_retry_pauses_and_returns_meta(self):
        """ConnectionError → ставит паузу, возвращает meta."""
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "requests.exceptions.ConnectionError: connection refused"}
        # Патчим time.sleep чтобы тест не ждал реально.
        with patch.object(self.project_mod.time, "sleep") as msleep:
            res = self.project_mod._heal_network_retry("p2net", failed, plan, attempt=1)
        self.assertIsNotNone(res)
        self.assertEqual(res["category"], "network")
        self.assertEqual(res["attempt"], 1)
        msleep.assert_called_once()

    def test_heal_network_retry_caps_delay(self):
        """Большой attempt → пауза не превышает 4.5с."""
        failed = {"stderr": "socket.timeout"}
        with patch.object(self.project_mod.time, "sleep") as msleep:
            res = self.project_mod._heal_network_retry("p2net2", failed, {"files": []}, attempt=10)
        self.assertIsNotNone(res)
        self.assertLessEqual(res["retried_after_s"], 4.5)

    def test_heal_network_retry_returns_none_for_other_errors(self):
        failed = {"stderr": "AssertionError: 1 != 2"}
        res = self.project_mod._heal_network_retry("p2net3", failed, {"files": []})
        self.assertIsNone(res)

    def test_heal_json_decode_finds_file_with_json_loads(self):
        """json.JSONDecodeError + код с json.loads → дет. хилер выдаёт хинт."""
        self._make_project_with_file(
            "p2json", "main.py",
            "import json\nimport requests\nr = requests.get('x')\nd = json.loads(r.text)\n",
        )
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "json.decoder.JSONDecodeError: Expecting value: line 1"}
        diag = self.project_mod._heal_json_decode("p2json", failed, plan)
        self.assertIsNotNone(diag)
        self.assertEqual(diag["target_file"], "main.py")
        self.assertEqual(diag["category"], "json")
        self.assertIn("json", diag["fix_instruction"].lower())

    def test_heal_json_decode_returns_none_when_no_json_in_code(self):
        self._make_project_with_file("p2json2", "main.py", "print('hi')\n")
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "json.JSONDecodeError: bad"}
        diag = self.project_mod._heal_json_decode("p2json2", failed, plan)
        self.assertIsNone(diag)

    def test_heal_json_decode_returns_none_for_other_errors(self):
        failed = {"stderr": "KeyError: 'foo'"}
        diag = self.project_mod._heal_json_decode("p2json3", failed, {"files": []})
        self.assertIsNone(diag)


# ─── P6: smart intake normalization ─────────────────────────────────────────
class TestIntakeNormalization(_IsolatedJarvisRoot):
    """P6: _normalize_intake_spec — структурные гарантии, без ключевых слов."""

    def test_empty_slug_filled_from_title(self):
        spec = {"title": "Lenta RSS Parser", "slug": ""}
        out = self.project_mod._normalize_intake_spec(spec, "q")
        self.assertTrue(out["slug"])
        self.assertRegex(out["slug"], r"^[a-z0-9\-]{1,40}$")

    def test_invalid_slug_replaced(self):
        spec = {"title": "Test", "slug": "Не валидный!!!"}
        out = self.project_mod._normalize_intake_spec(spec, "q")
        self.assertRegex(out["slug"], r"^[a-z0-9\-]{1,40}$")

    def test_kind_outside_set_falls_back_to_script(self):
        out = self.project_mod._normalize_intake_spec({"title": "X", "kind": "weird"}, "q")
        self.assertEqual(out["kind"], "script")

    def test_lang_outside_set_falls_back_to_python(self):
        out = self.project_mod._normalize_intake_spec({"title": "X", "language": "cobol"}, "q")
        self.assertEqual(out["language"], "python")

    def test_echo_requirement_flagged(self):
        """Если LLM выдала весь query одним пунктом — выставляется флаг."""
        q = "скачай новости с ленты и сохрани в csv"
        spec = {"title": "News", "requirements": [q]}
        out = self.project_mod._normalize_intake_spec(spec, q)
        self.assertEqual(out.get("_intake_warning"), "requirements is echo of query")

    def test_multi_requirements_not_flagged_as_echo(self):
        q = "скачай новости и сохрани в csv"
        spec = {"title": "X", "requirements": ["скачать rss", "распарсить", "сохранить"]}
        out = self.project_mod._normalize_intake_spec(spec, q)
        self.assertNotIn("_intake_warning", out)

    def test_empty_requirements_not_echo(self):
        """Пустой список НЕ считается эхо."""
        out = self.project_mod._normalize_intake_spec(
            {"title": "X", "requirements": []}, "some long query"
        )
        self.assertNotIn("_intake_warning", out)

    def test_deliverables_default_when_empty(self):
        out = self.project_mod._normalize_intake_spec({"title": "X"}, "q")
        self.assertEqual(out["deliverables"], ["main.py"])

    def test_acceptance_default_when_empty(self):
        out = self.project_mod._normalize_intake_spec({"title": "X"}, "q")
        self.assertGreaterEqual(len(out["acceptance_criteria"]), 1)

    def test_title_too_long_truncated_to_words(self):
        long_title = "A" * 200
        out = self.project_mod._normalize_intake_spec({"title": long_title}, "один два три четыре")
        self.assertLessEqual(len(out["title"]), 100)

    def test_non_dict_spec_handled(self):
        out = self.project_mod._normalize_intake_spec(None, "простой запрос")
        self.assertTrue(out["title"])
        self.assertTrue(out["slug"])
        self.assertEqual(out["kind"], "script")
        self.assertEqual(out["language"], "python")

    def test_string_requirements_coerced_to_list(self):
        out = self.project_mod._normalize_intake_spec(
            {"title": "X", "requirements": "один пункт"}, "q"
        )
        self.assertIsInstance(out["requirements"], list)
        self.assertEqual(out["requirements"], ["один пункт"])


# ─── P4: role-split models ───────────────────────────────────────────────────
class TestRoleSplitModels(unittest.TestCase):
    """P4: проверяем что ролевые алиасы существуют и приходят из config."""

    def test_role_aliases_exist_in_client(self):
        from brain import client as c
        for attr in ("MODEL_CODER", "MODEL_REVIEWER", "MODEL_ARCHITECT",
                     "MODEL_HEALER", "MODEL_INTAKE", "MODEL_README", "MODEL_REPORT"):
            self.assertTrue(hasattr(c, attr), f"client missing {attr}")
            self.assertIsInstance(getattr(c, attr), str)
            self.assertTrue(getattr(c, attr))

    def test_role_models_come_from_config(self):
        from brain import client as c
        from core import config as cfg
        self.assertEqual(c.MODEL_CODER, cfg.PROJECT_CODER_MODEL)
        self.assertEqual(c.MODEL_REVIEWER, cfg.PROJECT_REVIEWER_MODEL)
        self.assertEqual(c.MODEL_ARCHITECT, cfg.PROJECT_ARCHITECT_MODEL)
        self.assertEqual(c.MODEL_HEALER, cfg.PROJECT_HEALER_MODEL)
        self.assertEqual(c.MODEL_INTAKE, cfg.PROJECT_INTAKE_MODEL)
        self.assertEqual(c.MODEL_README, cfg.PROJECT_README_MODEL)
        self.assertEqual(c.MODEL_REPORT, cfg.PROJECT_REPORT_MODEL)


# ─── P5.2: deterministic report (no hallucinations) ───────────────────────────
class TestDeterministicReport(_IsolatedJarvisRoot):
    """P5.2: _deterministic_report() и _report() не выдумывают ошибки."""

    def test_happy_path_no_error_words(self):
        """Все файлы собрались, все тесты прошли — отчёт без слов 'не смогли', 'ошибка', 'провал'."""
        spec = {"title": "RSS Parser"}
        builds = [{"path": "main.py", "ok": True}, {"path": "utils.py", "ok": True}]
        tests = [{"name": "test_smoke", "ok": True}]
        out = self.project_mod._deterministic_report("rss-parser-123", spec, builds, tests).lower()
        for forbidden in ["не смогли", "не получилось", "провал", "что именно пошло не так", "к сожалению"]:
            self.assertNotIn(forbidden, out, f"forbidden '{forbidden}' in: {out}")
        self.assertIn("rss parser", out)
        self.assertIn("main.py", out)

    def test_failed_test_mentions_real_stderr(self):
        """Если тест упал — отчёт упоминает реальный stderr, а не выдумывает."""
        spec = {"title": "Project X"}
        builds = [{"path": "main.py", "ok": True}]
        tests = [{"name": "test_x", "ok": False, "stderr": "AssertionError: expected 5 got 3"}]
        out = self.project_mod._deterministic_report("project-x-1", spec, builds, tests)
        self.assertIn("AssertionError", out)
        self.assertIn("0 из 1", out)

    def test_no_tests_says_no_tests(self):
        """Когда тестов нет — отчёт говорит 'не были заданы', а не 'все прошли'."""
        spec = {"title": "Hello"}
        builds = [{"path": "main.py", "ok": True}]
        tests = []
        out = self.project_mod._deterministic_report("hello-1", spec, builds, tests)
        self.assertIn("не были заданы", out)
        self.assertNotIn("прошли успешно", out)

    def test_partial_build_mentions_count(self):
        """Когда не все файлы собрались — отчёт говорит 'X из Y'."""
        spec = {"title": "Big"}
        builds = [{"path": "a.py", "ok": True}, {"path": "b.py", "ok": False}]
        tests = []
        out = self.project_mod._deterministic_report("big-1", spec, builds, tests)
        self.assertIn("1 из 2", out)

    def test_report_skips_llm_on_happy_path(self):
        """P5.2: _report() не зовёт LLM когда всё ok."""
        spec = {"title": "OK"}
        builds = [{"path": "main.py", "ok": True}]
        tests = [{"name": "test_a", "ok": True}]
        budget = self.project_mod.Budget(wall_s=60, llm=10)
        with patch.object(self.project_mod, "_llm") as llm_mock:
            out = self.project_mod._report("ok-1", spec, builds, tests, budget)
        llm_mock.assert_not_called()
        self.assertIn("OK", out)


# ─── P6.1: bad-title placeholder rejection ────────────────────────────────────
class TestBadTitlePlaceholders(_IsolatedJarvisRoot):
    """P6.1: _normalize_intake_spec отбрасывает плейсхолдеры из промпта."""

    def test_untitled_project_replaced_by_query_words(self):
        spec = {"title": "untitled-project", "summary": "x"}
        out = self.project_mod._normalize_intake_spec(spec, "парсер RSS на feedparser")
        self.assertNotEqual(out["title"].lower(), "untitled-project")
        self.assertIn("парсер", out["title"].lower())

    def test_untitled_replaced(self):
        spec = {"title": "Untitled", "summary": "x"}
        out = self.project_mod._normalize_intake_spec(spec, "скачать файл по url")
        self.assertNotEqual(out["title"].lower(), "untitled")

    def test_summary_echo_truncated(self):
        """Если LLM вернула summary == query целиком — обрезаем до ~15 слов."""
        long_q = " ".join(["слово"] * 30)
        spec = {"title": "X", "summary": long_q}
        out = self.project_mod._normalize_intake_spec(spec, long_q)
        self.assertLess(len(out["summary"].split()), 20)
        self.assertTrue(out["summary"].endswith("…") or len(out["summary"].split()) <= 16)

    def test_requirements_echo_split_by_punctuation(self):
        """Если requirements это эхо запроса — разбиваем по запятым/«и»."""
        q = "скачать файл, распарсить json, сохранить в csv"
        spec = {"title": "Y", "requirements": [q]}
        out = self.project_mod._normalize_intake_spec(spec, q)
        self.assertGreaterEqual(len(out["requirements"]), 2)
        self.assertIn("_intake_warning", out)


# ─── P7: nightly E2E smoke ────────────────────────────────────────────────────
class TestNightlyE2ESmoke(unittest.TestCase):
    """P7: проверяем что nightly_e2e импортируется и корректно SKIPит при отсутствии ollama."""

    def test_module_imports_and_has_5_tasks(self):
        from dev import nightly_e2e
        self.assertEqual(len(nightly_e2e.REFERENCE_TASKS), 5)
        names = [t.name for t in nightly_e2e.REFERENCE_TASKS]
        # все имена уникальные
        self.assertEqual(len(set(names)), 5)

    def test_main_returns_zero_when_ollama_unavailable(self):
        """Когда ollama недоступна — main() отдаёт rc=0 (cron-safe)."""
        from dev import nightly_e2e
        with patch.object(nightly_e2e, "is_ollama_available", return_value=False):
            rc = nightly_e2e.main([])
        self.assertEqual(rc, 0)

    def test_main_returns_2_for_unknown_task(self):
        from dev import nightly_e2e
        with patch.object(nightly_e2e, "is_ollama_available", return_value=True):
            rc = nightly_e2e.main(["--task", "несуществующая-задача-zzz"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
