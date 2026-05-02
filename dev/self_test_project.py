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

    # ─── P11.0.1: shell-команды и pipe в тестах проекта ───────────────────
    def test_has_shell_metachars_detects_pipe_redirect_chain(self):
        """Детектор не реагирует на обычные команды, ловит |, <, >, &&, ||."""
        h = self.proj_mod._has_shell_metachars
        self.assertFalse(h("python main.py"))
        self.assertFalse(h("pytest tests/"))
        self.assertFalse(h(""))
        self.assertTrue(h("echo /list | python main.py"))
        self.assertTrue(h("python main.py < input.txt"))
        self.assertTrue(h("python main.py > output.txt"))
        self.assertTrue(h("python a.py && python b.py"))
        self.assertTrue(h("python a.py || echo failed"))

    def test_normalize_python_in_shell_cmd_replaces_python_token(self):
        """'python ...' в начале и после разделителей заменяется на sys.executable."""
        norm = self.proj_mod._normalize_python_in_shell_cmd
        out = norm("echo X | python main.py")
        # python должен быть заменён на полный путь
        self.assertIn(sys.executable.replace("\\", "\\"), out.replace('"', ''))
        self.assertNotIn(" python ", " " + out + " ")  # голого python не осталось
        # python3.exe тоже хватается
        out2 = norm("python3 main.py")
        self.assertNotIn("python3 ", out2)
        # Но слово внутри аргумента (в середине) не трогается
        out3 = norm("echo 'use python carefully'")
        self.assertIn("python", out3)  # оставили как было

    def test_run_shell_in_project_pipes_stdin_to_python(self):
        """Пайп входа в python -c — это ровно тот кейс который ломался в P11.0."""
        m = self.proj_mod.create_project({"title": "PIPE", "slug": "pipe-test"})
        # Пишем мини-скрипт который читает stdin
        self.proj_mod.write_project_file(
            m.slug, "reader.py",
            "import sys\ndata = sys.stdin.read().strip()\nprint(f'GOT:{data}')\n"
        )
        # На Windows echo встраивает в cmd.exe; на POSIX — встроенная shell builtin.
        # В обоих случаях работает через shell=True.
        res = self.proj_mod.run_shell_in_project(m.slug, "echo hello | python reader.py")
        self.assertEqual(res["returncode"], 0, msg=f"rc={res['returncode']} stderr={res.get('stderr')!r}")
        self.assertIn("GOT:hello", res["stdout"])
        # в нормализованной команде должен быть sys.executable
        self.assertNotIn(" python ", " " + res["normalized_cmd"] + " ")

    def test_run_shell_rejects_empty_cmd(self):
        m = self.proj_mod.create_project({"title": "E0", "slug": "empty-shell"})
        with self.assertRaises(ValueError):
            self.proj_mod.run_shell_in_project(m.slug, "")
        with self.assertRaises(ValueError):
            self.proj_mod.run_shell_in_project(m.slug, "   ")

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
        # P9.6: create_project теперь всегда добавляет ts-suffix — используем m.slug.
        m = self._make_project_with_file("p2syn", "main.py", "def f(:\n    pass\n")
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "  File \"main.py\", line 1\nSyntaxError: invalid syntax\n"}
        diag = self.project_mod._heal_syntax_error(m.slug, failed, plan)
        self.assertIsNotNone(diag)
        self.assertEqual(diag["target_file"], "main.py")
        self.assertEqual(diag["category"], "syntax")
        self.assertTrue(diag["fix_instruction"])

    def test_heal_syntax_error_returns_none_when_stderr_clean(self):
        """Без SyntaxError в stderr — None."""
        m = self._make_project_with_file("p2syn2", "main.py", "print('ok')\n")
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "AssertionError: not equal"}
        diag = self.project_mod._heal_syntax_error(m.slug, failed, plan)
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
        m = self._make_project_with_file(
            "p2json", "main.py",
            "import json\nimport requests\nr = requests.get('x')\nd = json.loads(r.text)\n",
        )
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "json.decoder.JSONDecodeError: Expecting value: line 1"}
        diag = self.project_mod._heal_json_decode(m.slug, failed, plan)
        self.assertIsNotNone(diag)
        self.assertEqual(diag["target_file"], "main.py")
        self.assertEqual(diag["category"], "json")
        self.assertIn("json", diag["fix_instruction"].lower())

    def test_heal_json_decode_returns_none_when_no_json_in_code(self):
        m = self._make_project_with_file("p2json2", "main.py", "print('hi')\n")
        plan = {"files": [{"path": "main.py"}]}
        failed = {"stderr": "json.JSONDecodeError: bad"}
        diag = self.project_mod._heal_json_decode(m.slug, failed, plan)
        self.assertIsNone(diag)

    def test_heal_missing_module_recognizes_bs4_feature_not_found(self):
        """P9.6: bs4.FeatureNotFound: features you requested: xml → pip install lxml."""
        m = self._make_project_with_file("p2bs4", "main.py", "x=1\n")
        stderr = (
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 1, in <module>\n"
            "    soup = BeautifulSoup(c, 'xml')\n"
            "bs4.exceptions.FeatureNotFound: Couldn't find a tree builder with the features you requested: xml. "
            "Do you need to install a parser library?\n"
        )
        with patch.object(self.project_mod, "pip_install", return_value={"ok": True}) as mock_pip:
            res = self.project_mod._heal_missing_module(m.slug, {"stderr": stderr})
        self.assertIsNotNone(res)
        self.assertTrue(res["ok"])
        self.assertEqual(res["installed_as"], "lxml")
        mock_pip.assert_called_once_with(m.slug, ["lxml"])

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


# ─── P9: aider build path switching ───────────────────────────────────
class TestAiderBuildPathSwitch(_IsolatedJarvisRoot):
    """P9: развилка _build_one_file (aider vs legacy) работает правильно."""

    def _make_target_and_budget(self):
        target = {"path": "main.py", "purpose": "hello world"}
        budget = self.project_mod.Budget(wall_s=60, llm=10)
        return target, budget

    def _make_project(self, slug: str = "aider-test"):
        return self.proj_mod.create_project({"title": "X", "slug": slug})

    def test_aider_used_when_enabled_and_available(self):
        """AIDER_ENABLED=True + aider доступен → вызывается _build_one_file_aider."""
        m = self._make_project()
        target, budget = self._make_target_and_budget()
        spec = {"title": "X", "summary": "hello", "requirements": ["печатай hi"]}
        plan = {"files": [target]}

        from core import config as cfg
        with patch.object(cfg, "AIDER_ENABLED", True), \
             patch.object(self.project_mod.aider_runner, "is_aider_available", return_value=True), \
             patch.object(self.project_mod, "_build_one_file_aider",
                          return_value={"path": "main.py", "ok": True, "_via": "aider",
                                        "verdict": "approve", "iters": 1, "static": {}}) as mock_aider:
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)
        mock_aider.assert_called_once()
        self.assertEqual(res["_via"], "aider")
        self.assertTrue(res["ok"])

    def test_legacy_used_when_aider_disabled(self):
        """AIDER_ENABLED=False → _build_one_file_aider НЕ вызывается, идёт старый путь."""
        m = self._make_project()
        target, budget = self._make_target_and_budget()
        spec = {"title": "X", "summary": "hello", "requirements": ["x"]}
        plan = {"files": [target]}

        from core import config as cfg
        with patch.object(cfg, "AIDER_ENABLED", False), \
             patch.object(self.project_mod, "_build_one_file_aider") as mock_aider, \
             patch.object(self.project_mod.coder_agent, "write_file", return_value="print('hi')\n"), \
             patch.object(self.project_mod.reviewer_agent, "review",
                          return_value={"verdict": "approve", "issues": [], "summary": "ok", "_source": "static"}):
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)
        mock_aider.assert_not_called()
        self.assertTrue(res["ok"])
        self.assertNotIn("_via", res)  # legacy не ставит _via

    def test_legacy_used_when_aider_missing_binary(self):
        """AIDER_ENABLED=True но is_aider_available()=False → fallback на legacy."""
        m = self._make_project()
        target, budget = self._make_target_and_budget()
        spec = {"title": "X", "summary": "hello", "requirements": ["x"]}
        plan = {"files": [target]}

        from core import config as cfg
        with patch.object(cfg, "AIDER_ENABLED", True), \
             patch.object(self.project_mod.aider_runner, "is_aider_available", return_value=False), \
             patch.object(self.project_mod, "_build_one_file_aider") as mock_aider, \
             patch.object(self.project_mod.coder_agent, "write_file", return_value="x=1\n"), \
             patch.object(self.project_mod.reviewer_agent, "review",
                          return_value={"verdict": "approve", "issues": [], "summary": "ok", "_source": "static"}):
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)
        mock_aider.assert_not_called()
        self.assertTrue(res["ok"])

    def test_aider_exception_falls_back_to_legacy(self):
        """Если _build_one_file_aider кидает любое исключение (кроме BudgetExceeded) → fallback."""
        m = self._make_project()
        target, budget = self._make_target_and_budget()
        spec = {"title": "X", "summary": "hello", "requirements": ["x"]}
        plan = {"files": [target]}

        from core import config as cfg
        with patch.object(cfg, "AIDER_ENABLED", True), \
             patch.object(self.project_mod.aider_runner, "is_aider_available", return_value=True), \
             patch.object(self.project_mod, "_build_one_file_aider",
                          side_effect=RuntimeError("aider exploded")), \
             patch.object(self.project_mod.coder_agent, "write_file", return_value="x=1\n"), \
             patch.object(self.project_mod.reviewer_agent, "review",
                          return_value={"verdict": "approve", "issues": [], "summary": "ok", "_source": "static"}):
            res = self.project_mod._build_one_file(m.slug, spec, plan, target, budget)
        self.assertTrue(res["ok"])  # legacy довёл до конца

    def test_budget_exceeded_in_aider_propagates(self):
        """BudgetExceeded из aider-пути НЕ перехватывается — всегда всплывает наверх."""
        m = self._make_project()
        target, budget = self._make_target_and_budget()
        spec = {"title": "X"}
        plan = {"files": [target]}

        from core import config as cfg
        with patch.object(cfg, "AIDER_ENABLED", True), \
             patch.object(self.project_mod.aider_runner, "is_aider_available", return_value=True), \
             patch.object(self.project_mod, "_build_one_file_aider",
                          side_effect=self.project_mod.BudgetExceeded("out of budget")):
            with self.assertRaises(self.project_mod.BudgetExceeded):
                self.project_mod._build_one_file(m.slug, spec, plan, target, budget)


# ─── P7: nightly E2E smoke ─────────────────────────────────────────────────
class TestHealViaAider(_IsolatedJarvisRoot):
    """P9.4: aider-ветка в _heal_loop работает корректно."""

    def test_pick_heal_target_finds_file_in_stderr(self):
        # P11.4: реальный Python traceback использует двойные кавычки в frame-строке
        plan = {"files": [{"path": "main.py"}, {"path": "helper.py"}]}
        failed = {"stderr": 'Traceback ...\n  File "helper.py", line 5, in <module>\nValueError'}
        self.assertEqual(self.project_mod._pick_heal_target(plan, failed), "helper.py")

    def test_pick_heal_target_falls_back_to_first_py(self):
        plan = {"files": [{"path": "data.csv"}, {"path": "main.py"}, {"path": "helper.py"}]}
        failed = {"stderr": "some unrelated error without filenames"}
        self.assertEqual(self.project_mod._pick_heal_target(plan, failed), "main.py")

    def test_pick_heal_target_returns_none_for_empty_plan(self):
        self.assertIsNone(self.project_mod._pick_heal_target({"files": []}, {"stderr": "x"}))

    def test_heal_via_aider_success(self):
        m = self.proj_mod.create_project({"title": "X", "slug": "heal-ok"})
        plan = {"files": [{"path": "main.py"}]}
        failed = {"name": "smoke", "stderr": "NameError", "command": "python main.py"}
        fake_result = type("R", (), {
            "ok": True, "error": "", "duration_s": 1.5, "attempts": 1, "content": "x=1\n",
            "stderr": "", "exit_code": 0,
        })()
        with patch.object(self.project_mod.aider_runner, "aider_heal", return_value=fake_result) as mock_heal:
            res = self.project_mod._heal_via_aider(m.slug, plan, failed)
        mock_heal.assert_called_once()
        self.assertTrue(res["ok"])
        self.assertEqual(res["target"], "main.py")

    def test_heal_via_aider_handles_exception(self):
        m = self.proj_mod.create_project({"title": "X", "slug": "heal-ex"})
        plan = {"files": [{"path": "main.py"}]}
        failed = {"name": "smoke", "stderr": "x", "command": "python main.py"}
        with patch.object(self.project_mod.aider_runner, "aider_heal", side_effect=RuntimeError("boom")):
            res = self.project_mod._heal_via_aider(m.slug, plan, failed)
        self.assertFalse(res["ok"])
        self.assertIn("boom", res["error"])

    def test_heal_loop_uses_aider_when_enabled(self):
        m = self.proj_mod.create_project({"title": "X", "slug": "heal-aider"})
        self.proj_mod.write_project_file(m.slug, "main.py", "x=1\n")
        plan = {"files": [{"path": "main.py", "purpose": "entry"}],
                "tests": [{"name": "smoke", "command": "python main.py", "checks": [{"type": "rc_zero"}]}]}
        spec = {"title": "X", "summary": "", "requirements": []}
        budget = self.project_mod.Budget(wall_s=60, llm=10)
        test_results_failing = [{"name": "smoke", "ok": False, "stderr": "some runtime err",
                                  "stdout": "", "command": "python main.py", "rc": 1}]
        test_results_ok = [{"name": "smoke", "ok": True, "stderr": "", "stdout": "",
                            "command": "python main.py", "rc": 0}]
        from core import config as cfg
        fake_ah = {"ok": True, "target": "main.py", "error": "", "duration_s": 2.0, "attempts": 1}
        with patch.object(cfg, "AIDER_ENABLED", True), \
             patch.object(self.project_mod.aider_runner, "is_aider_available", return_value=True), \
             patch.object(self.project_mod, "_heal_via_aider", return_value=fake_ah) as mock_aider, \
             patch.object(self.project_mod, "_test", return_value=test_results_ok), \
             patch.object(self.project_mod, "_diagnose") as mock_diagnose:
            out = self.project_mod._heal_loop(m.slug, spec, plan, test_results_failing, budget)
        mock_aider.assert_called_once()
        mock_diagnose.assert_not_called()
        self.assertTrue(all(r["ok"] for r in out))

    def test_heal_loop_falls_back_to_llm_when_aider_fails(self):
        m = self.proj_mod.create_project({"title": "X", "slug": "heal-fb"})
        self.proj_mod.write_project_file(m.slug, "main.py", "x=1\n")
        plan = {"files": [{"path": "main.py", "purpose": "entry"}],
                "tests": [{"name": "smoke", "command": "python main.py", "checks": [{"type": "rc_zero"}]}]}
        spec = {"title": "X", "summary": "", "requirements": []}
        budget = self.project_mod.Budget(wall_s=60, llm=10)
        test_results_failing = [{"name": "smoke", "ok": False, "stderr": "err",
                                  "stdout": "", "command": "python main.py", "rc": 1}]
        test_results_ok = [{"name": "smoke", "ok": True, "stderr": "", "stdout": "",
                            "command": "python main.py", "rc": 0}]
        from core import config as cfg
        fake_ah = {"ok": False, "target": "main.py", "error": "aider timeout"}
        with patch.object(cfg, "AIDER_ENABLED", True), \
             patch.object(self.project_mod.aider_runner, "is_aider_available", return_value=True), \
             patch.object(self.project_mod, "_heal_via_aider", return_value=fake_ah), \
             patch.object(self.project_mod, "_diagnose",
                          return_value={"target_file": "main.py", "diagnosis": "x", "fix_instruction": "fix it"}) as mock_diag, \
             patch.object(self.project_mod.coder_agent, "patch_file", return_value="y=2\n"), \
             patch.object(self.project_mod, "_test", return_value=test_results_ok):
            out = self.project_mod._heal_loop(m.slug, spec, plan, test_results_failing, budget)
        mock_diag.assert_called_once()
        self.assertTrue(all(r["ok"] for r in out))

    def test_heal_loop_skips_aider_when_disabled(self):
        m = self.proj_mod.create_project({"title": "X", "slug": "heal-off"})
        self.proj_mod.write_project_file(m.slug, "main.py", "x=1\n")
        plan = {"files": [{"path": "main.py"}],
                "tests": [{"name": "smoke", "command": "python main.py", "checks": [{"type": "rc_zero"}]}]}
        spec = {"title": "X"}
        budget = self.project_mod.Budget(wall_s=60, llm=10)
        test_results_failing = [{"name": "smoke", "ok": False, "stderr": "err",
                                  "stdout": "", "command": "python main.py", "rc": 1}]
        test_results_ok = [{"name": "smoke", "ok": True, "stderr": "", "stdout": "",
                            "command": "python main.py", "rc": 0}]
        from core import config as cfg
        with patch.object(cfg, "AIDER_ENABLED", False), \
             patch.object(self.project_mod, "_heal_via_aider") as mock_aider, \
             patch.object(self.project_mod, "_diagnose",
                          return_value={"target_file": "main.py", "diagnosis": "x", "fix_instruction": "fix"}), \
             patch.object(self.project_mod.coder_agent, "patch_file", return_value="x=2\n"), \
             patch.object(self.project_mod, "_test", return_value=test_results_ok):
            self.project_mod._heal_loop(m.slug, spec, plan, test_results_failing, budget)
        mock_aider.assert_not_called()


# ─── P9.7: фиксы плохих планов архитектора ─────────────────────────────────────
class TestPlanRobustnessP97(_IsolatedJarvisRoot):
    """P9.7: автофикстуры входов, мягкий file_min_size, изоляция deliverables."""

    # ─── _is_input_fixture ──────────────────────────────────────────────────────
    def test_is_input_fixture_true_for_unknown_data_file(self):
        # backup_example.txt не в deliverables, не исходник → это вход.
        self.assertTrue(self.project_mod._is_input_fixture("backup_example.txt", ["main.py", "output.csv"]))

    def test_is_input_fixture_false_for_deliverable(self):
        # output.csv в deliverables → это выход, не вход.
        self.assertFalse(self.project_mod._is_input_fixture("output.csv", ["main.py", "output.csv"]))

    def test_is_input_fixture_false_for_source_file(self):
        # main.py — исходник, никогда не dummy.
        self.assertFalse(self.project_mod._is_input_fixture("helper.py", []))

    def test_is_input_fixture_normalizes_slashes_and_prefix(self):
        # "./data/in.txt" и "data\\in.txt" должны матчиться с "data/in.txt" в deliverables.
        self.assertFalse(self.project_mod._is_input_fixture("./data/in.txt", ["data/in.txt"]))
        self.assertFalse(self.project_mod._is_input_fixture("data\\in.txt", ["data/in.txt"]))

    def test_is_input_fixture_empty_path(self):
        self.assertFalse(self.project_mod._is_input_fixture("", ["main.py"]))

    # ─── _prepare_test_fixtures ────────────────────────────────────────────────
    def _make_proj(self, deliverables):
        m = self.proj_mod.create_project({"title": "t", "slug": "p97", "summary": "s",
                                          "deliverables": deliverables, "requirements": ["r"]})
        return m

    def test_prepare_fixtures_creates_dummy_for_input_file(self):
        m = self._make_proj(["main.py", "output.csv"])
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "rc_zero"},
            {"type": "file_exists", "path": "input.csv"},
        ]}]}
        spec = {"deliverables": ["main.py", "output.csv"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, ["input.csv"])
        # P9.10: для .csv создаётся РЕАЛИСТИЧНЫЙ sample (не пустой).
        p = self.proj_mod.safe_project_path(m.slug, "input.csv")
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 0, "P9.10: файл должен быть непустым")

    def test_prepare_fixtures_skips_deliverable(self):
        m = self._make_proj(["main.py", "output.csv"])
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "file_exists", "path": "output.csv"},
        ]}]}
        spec = {"deliverables": ["main.py", "output.csv"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, [])

    def test_prepare_fixtures_skips_source_file(self):
        m = self._make_proj(["main.py"])
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "file_exists", "path": "helper.py"},
        ]}]}
        spec = {"deliverables": ["main.py"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, [])

    def test_prepare_fixtures_no_spec_returns_empty(self):
        m = self._make_proj(["main.py"])
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "file_exists", "path": "x.txt"},
        ]}]}
        self.assertEqual(self.project_mod._prepare_test_fixtures(m.slug, plan, None), [])

    def test_prepare_fixtures_does_not_overwrite_existing(self):
        m = self._make_proj(["main.py"])
        # рукой создаём файл с содержимым
        p = self.proj_mod.safe_project_path(m.slug, "input.txt")
        p.write_text("hello", encoding="utf-8")
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "file_exists", "path": "input.txt"},
        ]}]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, {"deliverables": ["main.py"]})
        self.assertEqual(created, [])  # уже существует — не трогаем
        self.assertEqual(p.read_text(encoding="utf-8"), "hello")

    # ─── _normalize_min_size_check ────────────────────────────────────────────
    def test_normalize_min_size_softens_when_file_smaller_than_floor_demand(self):
        m = self._make_proj(["page.html"])
        p = self.proj_mod.safe_project_path(m.slug, "page.html")
        p.write_text("x" * 529, encoding="utf-8")  # как example.com
        check = {"type": "file_min_size", "path": "page.html", "bytes": 1024}
        out = self.project_mod._normalize_min_size_check(m.slug, check)
        self.assertTrue(out.get("_soft"))
        self.assertEqual(out.get("_original_bytes"), 1024)
        self.assertLessEqual(out.get("bytes"), 529)

    def test_normalize_min_size_keeps_strict_when_file_missing(self):
        m = self._make_proj(["page.html"])
        check = {"type": "file_min_size", "path": "page.html", "bytes": 1024}
        out = self.project_mod._normalize_min_size_check(m.slug, check)
        # файл не существует — check остаётся жёстким
        self.assertFalse(out.get("_soft"))
        self.assertEqual(out.get("bytes"), 1024)

    def test_normalize_min_size_keeps_strict_when_file_empty(self):
        m = self._make_proj(["page.html"])
        p = self.proj_mod.safe_project_path(m.slug, "page.html")
        p.write_bytes(b"")
        check = {"type": "file_min_size", "path": "page.html", "bytes": 1024}
        out = self.project_mod._normalize_min_size_check(m.slug, check)
        # файл пустой (< _MIN_SIZE_REALISTIC_FLOOR=64) — check остаётся жёстким
        self.assertFalse(out.get("_soft"))
        self.assertEqual(out.get("bytes"), 1024)

    def test_normalize_min_size_no_op_for_non_min_size_check(self):
        m = self._make_proj(["x"])
        check = {"type": "file_exists", "path": "foo.txt"}
        out = self.project_mod._normalize_min_size_check(m.slug, check)
        self.assertEqual(out, check)

    # ─── интеграция в _test ─────────────────────────────────────────────────────
    def test_test_function_filters_lonely_file_exists_p99(self):
        """P9.9 изменил поведение: одинокий file_exists на вход (без
        выходных чеков) убирается filter-этапом как бессмысленный.
        Тест проходит по оставшемуся rc_zero."""
        m = self._make_proj(["main.py"])
        self.proj_mod.write_project_file(m.slug, "main.py", "print('ok')\n")
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "rc_zero"},
            {"type": "file_exists", "path": "input_data.csv"},
        ]}]}
        spec = {"deliverables": ["main.py"]}
        results = self.project_mod._test(m.slug, plan, spec)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"], f"expected pass, got {results[0]}")

    def test_test_function_backward_compat_without_spec(self):
        m = self._make_proj(["main.py"])
        self.proj_mod.write_project_file(m.slug, "main.py", "print('ok')\n")
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "rc_zero"},
        ]}]}
        # без spec — работает как раньше
        results = self.project_mod._test(m.slug, plan)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"])


# ─── P9.9: output-aware fixtures ────────────────────────────────────
class TestOutputAwareFixturesP99(_IsolatedJarvisRoot):
    """P9.9: фикстуры НЕ создаются для путей, которые скрипт сам должен создать
    (путь упомянут в file_min_lines/file_min_size/json_valid/file_contains или
    в build_steps:create_file)."""

    def _make_proj(self, deliverables):
        m = self.proj_mod.create_project({"title": "t", "slug": "p99", "summary": "s",
                                          "deliverables": deliverables, "requirements": ["r"]})
        return m

    # ─── _collect_output_paths ──────────────────────────────────────
    def test_collect_output_paths_from_file_min_lines(self):
        plan = {"tests": [{"name": "s", "command": "python main.py", "checks": [
            {"type": "file_min_lines", "path": "emails.csv", "lines": 2},
        ]}]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertIn("emails.csv", out)

    def test_collect_output_paths_from_file_min_size(self):
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "file_min_size", "path": "page.html", "bytes": 100},
        ]}]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertIn("page.html", out)

    def test_collect_output_paths_from_json_valid(self):
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "json_valid", "path": "output.json"},
        ]}]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertIn("output.json", out)

    def test_collect_output_paths_from_file_contains(self):
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "file_contains", "path": "log.txt", "text": "abc"},
        ]}]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertIn("log.txt", out)

    def test_collect_output_paths_from_build_steps_create_file(self):
        plan = {"build_steps": [
            {"step": 1, "kind": "create_file", "target": "data.csv"},
        ]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertIn("data.csv", out)

    def test_collect_output_paths_normalizes_slashes(self):
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "file_min_lines", "path": "./out\\result.csv", "lines": 1},
        ]}]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertIn("out/result.csv", out)

    def test_collect_output_paths_ignores_file_exists(self):
        # file_exists САМ по себе не делает путь выходом — это может быть вход.
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "file_exists", "path": "input.csv"},
        ]}]}
        out = self.project_mod._collect_output_paths(plan)
        self.assertNotIn("input.csv", out)

    def test_collect_output_paths_empty_plan(self):
        self.assertEqual(self.project_mod._collect_output_paths({}), set())
        self.assertEqual(self.project_mod._collect_output_paths(None), set())

    # ─── _is_input_fixture с output_paths ─────────────────────────────
    def test_is_input_fixture_false_when_path_in_outputs(self):
        # emails.csv в output_paths → это выход, не вход.
        self.assertFalse(self.project_mod._is_input_fixture(
            "emails.csv", ["main.py"], output_paths={"emails.csv"}))

    def test_is_input_fixture_true_when_not_in_outputs(self):
        # input.csv НЕ в deliverables и НЕ в outputs → вход.
        self.assertTrue(self.project_mod._is_input_fixture(
            "input.csv", ["main.py"], output_paths={"emails.csv"}))

    def test_is_input_fixture_backward_compat_no_outputs(self):
        # Без output_paths (legacy) — работает как в P9.7.
        self.assertTrue(self.project_mod._is_input_fixture(
            "backup_example.txt", ["main.py"]))

    # ─── сценарии regex_extractor и rename_files (регрессии P9.8) ────────────────
    def test_prepare_fixtures_skips_emails_csv_when_file_min_lines(self):
        """Регрессия regex_extractor: emails.csv — ВЫХОД, фикстуру НЕ создавать."""
        m = self._make_proj(["main.py"])
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "rc_zero"},
            {"type": "file_exists", "path": "emails.csv"},
            {"type": "file_min_lines", "path": "emails.csv", "lines": 2},
        ]}]}
        spec = {"deliverables": ["main.py"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, [], "emails.csv — выход, фикстура не должна быть создана")
        # файл реально НЕ создан
        self.assertFalse(self.proj_mod.safe_project_path(m.slug, "emails.csv").exists())

    def test_prepare_fixtures_skips_path_in_create_file_step(self):
        """Регрессия rename_files: если путь создаётся в build_steps — это выход."""
        m = self._make_proj(["main.py"])
        plan = {
            "build_steps": [
                {"step": 1, "kind": "create_file", "target": "backup_example.txt"},
            ],
            "tests": [{"name": "smoke", "command": "python main.py", "checks": [
                {"type": "file_exists", "path": "backup_example.txt"},
            ]}],
        }
        spec = {"deliverables": ["main.py"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, [])
        self.assertFalse(self.proj_mod.safe_project_path(m.slug, "backup_example.txt").exists())

    # ─── _filter_invalid_checks ───────────────────────────────────
    def test_filter_removes_stdout_contains(self):
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "rc_zero"},
            {"type": "stdout_contains", "text": "Processing complete"},
        ]}]}
        new_plan, removed = self.project_mod._filter_invalid_checks(plan, {"deliverables": []})
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["reason"], "stdout_contains_violates_principles")
        types = [c["type"] for c in new_plan["tests"][0]["checks"]]
        self.assertNotIn("stdout_contains", types)
        self.assertIn("rc_zero", types)

    def test_filter_keeps_file_exists_on_output(self):
        """file_exists на выход (есть file_min_lines на тот же путь) ОСТАВЛЯЕМ."""
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "file_exists", "path": "emails.csv"},
            {"type": "file_min_lines", "path": "emails.csv", "lines": 2},
        ]}]}
        new_plan, removed = self.project_mod._filter_invalid_checks(plan, {"deliverables": []})
        self.assertEqual(removed, [])
        types = [c["type"] for c in new_plan["tests"][0]["checks"]]
        self.assertEqual(types.count("file_exists"), 1)
        self.assertEqual(types.count("file_min_lines"), 1)

    def test_filter_removes_file_exists_on_lonely_input(self):
        """file_exists на вход без выходных чеков — бессмыслен, убираем."""
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "rc_zero"},
            {"type": "file_exists", "path": "input.csv"},
        ]}]}
        new_plan, removed = self.project_mod._filter_invalid_checks(plan, {"deliverables": ["main.py"]})
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["reason"], "file_exists_on_input_fixture")
        types = [c["type"] for c in new_plan["tests"][0]["checks"]]
        self.assertEqual(types, ["rc_zero"])

    def test_filter_keeps_file_exists_on_deliverable(self):
        """file_exists на deliverable (выход явный) — ОСТАВЛЯЕМ."""
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "file_exists", "path": "output.json"},
        ]}]}
        new_plan, removed = self.project_mod._filter_invalid_checks(
            plan, {"deliverables": ["main.py", "output.json"]})
        self.assertEqual(removed, [])
        types = [c["type"] for c in new_plan["tests"][0]["checks"]]
        self.assertIn("file_exists", types)

    def test_filter_handles_none_spec(self):
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "stdout_contains", "text": "ok"},
            {"type": "rc_zero"},
        ]}]}
        new_plan, removed = self.project_mod._filter_invalid_checks(plan, None)
        self.assertEqual(len(removed), 1)
        types = [c["type"] for c in new_plan["tests"][0]["checks"]]
        self.assertEqual(types, ["rc_zero"])

    def test_filter_does_not_mutate_input_plan(self):
        """Исходный plan НЕ должен меняться по ссылке."""
        plan = {"tests": [{"name": "s", "command": "x", "checks": [
            {"type": "stdout_contains", "text": "x"},
            {"type": "rc_zero"},
        ]}]}
        original_len = len(plan["tests"][0]["checks"])
        self.project_mod._filter_invalid_checks(plan, None)
        self.assertEqual(len(plan["tests"][0]["checks"]), original_len)

    def test_prepare_fixtures_creates_only_real_input_when_mixed(self):
        """Смешанный сценарий: input.csv — вход, output.json — выход."""
        m = self._make_proj(["main.py"])
        plan = {"tests": [{"name": "smoke", "command": "python main.py", "checks": [
            {"type": "file_exists", "path": "input.csv"},      # вход → фикстура
            {"type": "file_exists", "path": "output.json"},     # выход, есть json_valid → нет фикстуры
            {"type": "json_valid", "path": "output.json"},
        ]}]}
        spec = {"deliverables": ["main.py"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, ["input.csv"])
        self.assertTrue(self.proj_mod.safe_project_path(m.slug, "input.csv").exists())
        self.assertFalse(self.proj_mod.safe_project_path(m.slug, "output.json").exists())


# ─── P9.10: realistic input fixtures + plan.inputs + heuristic ────────────────
class TestInputFixturesP10(_IsolatedJarvisRoot):
    """P9.10: реалистичные входы из plan.inputs[]/spec.summary."""

    def _make_proj(self, deliverables):
        m = self.proj_mod.create_project({"title": "t", "slug": "p10", "summary": "s",
                                          "deliverables": deliverables, "requirements": ["r"]})
        return m

    # ─── _default_sample_for ─────────────────────────────────────────
    def test_default_sample_for_txt_contains_email(self):
        sample = self.project_mod._default_sample_for("text.txt")
        self.assertIn(b"@", sample, "в .txt должен быть email для regex-тестов")
        self.assertGreater(len(sample), 50)

    def test_default_sample_for_csv_has_header_and_rows(self):
        sample = self.project_mod._default_sample_for("data.csv").decode("utf-8")
        lines = sample.splitlines()
        self.assertGreaterEqual(len(lines), 2, "CSV должен иметь заголовок + данные")
        self.assertIn(",", lines[0])

    def test_default_sample_for_json_is_valid(self):
        import json as _json
        sample = self.project_mod._default_sample_for("input.json").decode("utf-8")
        # реальный JSON должен парситься
        obj = _json.loads(sample)
        self.assertIsInstance(obj, dict)

    def test_default_sample_for_unknown_ext_returns_empty(self):
        self.assertEqual(self.project_mod._default_sample_for("unknown.xyz"), b"")

    def test_default_sample_for_normalizes_path(self):
        # backslashes и ./ префикс не ломают
        s1 = self.project_mod._default_sample_for("./data\\input.txt")
        self.assertGreater(len(s1), 0)

    # ─── _heuristic_input_paths ───────────────────────────────────
    def test_heuristic_finds_filename_in_summary(self):
        spec = {"summary": "ищет email-адреса в text.txt и сохраняет их в emails.csv",
                "deliverables": ["main.py", "emails.csv"]}
        found = self.project_mod._heuristic_input_paths(spec)
        self.assertIn("text.txt", found)
        # emails.csv — deliverable, исключается
        self.assertNotIn("emails.csv", found)

    def test_heuristic_skips_when_no_input_in_summary(self):
        spec = {"summary": "скачивает RSS и пишет в news.csv",
                "deliverables": ["main.py", "news.csv"]}
        found = self.project_mod._heuristic_input_paths(spec)
        # news.csv в deliverables — исключён, больше файлов нет
        self.assertEqual(found, [])

    def test_heuristic_empty_spec(self):
        self.assertEqual(self.project_mod._heuristic_input_paths(None), [])
        self.assertEqual(self.project_mod._heuristic_input_paths({}), [])
        self.assertEqual(self.project_mod._heuristic_input_paths({"summary": ""}), [])

    def test_heuristic_dedup(self):
        spec = {"summary": "читает data.csv в data.csv и data.csv", "deliverables": []}
        found = self.project_mod._heuristic_input_paths(spec)
        self.assertEqual(found, ["data.csv"])

    # ─── _collect_input_specs ───────────────────────────────────
    def test_collect_input_specs_plan_priority(self):
        plan = {"inputs": [{"path": "my_data.txt", "sample_content": "hello world"}]}
        spec = {"summary": "читает my_data.txt", "deliverables": ["main.py"]}
        out = self.project_mod._collect_input_specs(plan, spec)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["path"], "my_data.txt")
        self.assertEqual(out[0]["source"], "plan")
        self.assertEqual(out[0]["sample_content"], b"hello world")

    def test_collect_input_specs_heuristic_fallback(self):
        plan = {}  # архитектор забыл inputs
        spec = {"summary": "читает text.txt", "deliverables": ["main.py", "out.csv"]}
        out = self.project_mod._collect_input_specs(plan, spec)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["path"], "text.txt")
        self.assertEqual(out[0]["source"], "heuristic")
        self.assertGreater(len(out[0]["sample_content"]), 0)

    def test_collect_input_specs_dedup_plan_and_heuristic(self):
        plan = {"inputs": [{"path": "text.txt", "sample_content": "manual"}]}
        spec = {"summary": "читает text.txt", "deliverables": ["main.py"]}
        out = self.project_mod._collect_input_specs(plan, spec)
        # plan имеет приоритет, heuristic не дублирует
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["sample_content"], b"manual")
        self.assertEqual(out[0]["source"], "plan")

    def test_collect_input_specs_excludes_deliverables(self):
        plan = {"inputs": [{"path": "main.py", "sample_content": "x"}]}  # ошибка архитектора
        spec = {"deliverables": ["main.py"]}
        out = self.project_mod._collect_input_specs(plan, spec)
        self.assertEqual(out, [])

    def test_collect_input_specs_default_sample_when_missing(self):
        plan = {"inputs": [{"path": "data.json"}]}  # без sample_content
        spec = {"deliverables": []}
        out = self.project_mod._collect_input_specs(plan, spec)
        self.assertEqual(len(out), 1)
        # используется дефолтный sample для .json
        import json as _json
        obj = _json.loads(out[0]["sample_content"].decode("utf-8"))
        self.assertIsInstance(obj, dict)

    # ─── _prepare_test_fixtures с реалистичным входом ────────────
    def test_prepare_fixtures_uses_plan_inputs_with_realistic_content(self):
        """P9.10 сценарий regex_extractor: plan.inputs создаёт реалистичный text.txt."""
        m = self._make_proj(["main.py"])
        plan = {
            "inputs": [{"path": "text.txt", "sample_content":
                        "contact alice@example.com and bob@test.org"}],
            "tests": [{"name": "smoke", "command": "python main.py", "checks": [
                {"type": "rc_zero"},
                {"type": "file_min_lines", "path": "emails.csv", "lines": 2},
            ]}],
        }
        spec = {"deliverables": ["main.py"], "summary": "ищет email в text.txt"}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertIn("text.txt", created)
        p = self.proj_mod.safe_project_path(m.slug, "text.txt")
        self.assertTrue(p.exists())
        content = p.read_text(encoding="utf-8")
        self.assertIn("@", content, "в text.txt должен быть email")
        # emails.csv — выход, не должен создаваться
        self.assertFalse(self.proj_mod.safe_project_path(m.slug, "emails.csv").exists())

    def test_prepare_fixtures_uses_heuristic_when_plan_inputs_empty(self):
        """P9.10 регрессия: архитектор забыл plan.inputs — heuristic из summary."""
        m = self._make_proj(["main.py"])
        plan = {
            "tests": [{"name": "smoke", "command": "python main.py", "checks": [
                {"type": "rc_zero"},
                {"type": "file_min_lines", "path": "emails.csv", "lines": 2},
            ]}],
        }
        spec = {"deliverables": ["main.py"],
                "summary": "ищет email-адреса в text.txt и сохраняет в emails.csv"}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertIn("text.txt", created)
        p = self.proj_mod.safe_project_path(m.slug, "text.txt")
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 0, "дефолтный sample непустой")
        self.assertIn("@", p.read_text(encoding="utf-8"))

    def test_prepare_fixtures_no_overwrite_existing_input(self):
        m = self._make_proj(["main.py"])
        p = self.proj_mod.safe_project_path(m.slug, "text.txt")
        p.write_text("пользовательский контент", encoding="utf-8")
        plan = {"inputs": [{"path": "text.txt", "sample_content": "новый"}]}
        spec = {"deliverables": ["main.py"]}
        created = self.project_mod._prepare_test_fixtures(m.slug, plan, spec)
        self.assertEqual(created, [])
        self.assertEqual(p.read_text(encoding="utf-8"), "пользовательский контент")

    # ─── _enrich_plan_with_heuristic_inputs ─────────────────────────
    def test_enrich_adds_heuristic_inputs_when_plan_empty(self):
        plan = {"files": [{"path": "main.py"}]}
        spec = {"summary": "читает text.txt", "deliverables": ["main.py"]}
        out = self.project_mod._enrich_plan_with_heuristic_inputs(plan, spec)
        self.assertEqual(len(out["inputs"]), 1)
        self.assertEqual(out["inputs"][0]["path"], "text.txt")
        self.assertEqual(out["inputs"][0]["_source"], "heuristic")
        self.assertGreater(len(out["inputs"][0]["sample_content"]), 0)

    def test_enrich_does_not_override_architect_inputs(self):
        plan = {"inputs": [{"path": "text.txt", "sample_content": "архитекторский"}]}
        spec = {"summary": "читает text.txt", "deliverables": ["main.py"]}
        out = self.project_mod._enrich_plan_with_heuristic_inputs(plan, spec)
        self.assertEqual(len(out["inputs"]), 1)
        self.assertEqual(out["inputs"][0]["sample_content"], "архитекторский")

    def test_enrich_skips_deliverables(self):
        plan = {}
        spec = {"summary": "пишет в output.csv", "deliverables": ["main.py", "output.csv"]}
        out = self.project_mod._enrich_plan_with_heuristic_inputs(plan, spec)
        # output.csv — deliverable, не должен попасть в inputs
        self.assertEqual(out.get("inputs", []), [])

    def test_enrich_handles_none_or_empty(self):
        # None plan — не падаем
        self.assertIsNone(self.project_mod._enrich_plan_with_heuristic_inputs(None, {"summary": "x text.txt"}))
        # пустой spec — ничего не добавляет
        plan = {"files": []}
        out = self.project_mod._enrich_plan_with_heuristic_inputs(plan, None)
        self.assertNotIn("inputs", out)


class TestPlanContractsP11(unittest.TestCase):
    """P11.1: валидация и нормализация exports/depends_on в plan.files."""

    @classmethod
    def setUpClass(cls):
        import brain.agents.project as project_mod
        cls.pm = project_mod

    # --- _normalize_export_entry ---

    def test_normalize_export_rejects_non_dict(self):
        self.assertIsNone(self.pm._normalize_export_entry(None))
        self.assertIsNone(self.pm._normalize_export_entry("name"))
        self.assertIsNone(self.pm._normalize_export_entry([]))

    def test_normalize_export_rejects_invalid_name(self):
        # пустое имя
        self.assertIsNone(self.pm._normalize_export_entry({"name": ""}))
        # имя с пробелом
        self.assertIsNone(self.pm._normalize_export_entry({"name": "bad name"}))
        # имя начинающееся с цифры
        self.assertIsNone(self.pm._normalize_export_entry({"name": "1foo"}))

    def test_normalize_export_kind_heuristic_const(self):
        out = self.pm._normalize_export_entry({"name": "MAX_LIMIT"})
        self.assertEqual(out["kind"], "const")

    def test_normalize_export_kind_heuristic_class(self):
        out = self.pm._normalize_export_entry({"name": "UserService"})
        self.assertEqual(out["kind"], "class")

    def test_normalize_export_kind_heuristic_function(self):
        out = self.pm._normalize_export_entry({"name": "add_user"})
        self.assertEqual(out["kind"], "function")

    def test_normalize_export_kind_explicit_wins(self):
        # Архитектор явно указал kind — мы не переопределяем эвристикой.
        out = self.pm._normalize_export_entry({"name": "User", "kind": "function"})
        self.assertEqual(out["kind"], "function")

    def test_normalize_export_invalid_kind_falls_back(self):
        out = self.pm._normalize_export_entry({"name": "add_user", "kind": "мусор"})
        # неизвестный kind → эвристика по виду имени
        self.assertEqual(out["kind"], "function")

    def test_normalize_export_keeps_signature_and_doc(self):
        out = self.pm._normalize_export_entry({
            "name": "add", "signature": "(a: int, b: int) -> int", "doc": "Sum."
        })
        self.assertEqual(out["signature"], "(a: int, b: int) -> int")
        self.assertEqual(out["doc"], "Sum.")

    # --- _file_likely_entry_point ---

    def test_entry_point_main_py(self):
        self.assertTrue(self.pm._file_likely_entry_point({"path": "main.py"}))

    def test_entry_point_app_py(self):
        self.assertTrue(self.pm._file_likely_entry_point({"path": "src/app.py"}))

    def test_entry_point_storage_is_not(self):
        self.assertFalse(self.pm._file_likely_entry_point({"path": "storage.py"}))

    def test_entry_point_non_dict(self):
        self.assertFalse(self.pm._file_likely_entry_point(None))
        self.assertFalse(self.pm._file_likely_entry_point("main.py"))

    # --- _is_python_path / _is_non_python_path ---

    def test_is_python_path(self):
        self.assertTrue(self.pm._is_python_path("foo.py"))
        self.assertTrue(self.pm._is_python_path("src/utils.py"))
        self.assertFalse(self.pm._is_python_path("data.json"))
        self.assertFalse(self.pm._is_python_path("README.md"))

    # --- _normalize_plan_contracts ---

    def test_normalize_plan_contracts_lossless_on_garbage(self):
        # Не падает на мусорных входах
        self.assertEqual(self.pm._normalize_plan_contracts(None, None), None)
        out = self.pm._normalize_plan_contracts({}, None)
        self.assertIsInstance(out, dict)

    def test_normalize_plan_contracts_fills_metrics(self):
        plan = {
            "files": [
                {"path": "main.py", "role": "entry", "depends_on": ["storage.py"], "exports": []},
                {"path": "storage.py", "role": "data", "depends_on": [],
                 "exports": [{"name": "add_user", "signature": "(name: str)"}]},
            ]
        }
        out = self.pm._normalize_plan_contracts(plan, None)
        m = out["_contract_metrics"]
        self.assertEqual(m["files_total"], 2)
        self.assertEqual(m["py_files"], 2)
        self.assertEqual(m["files_with_exports"], 1)
        # main.py — entry-point, отсутствие exports допустимо
        self.assertEqual(m["files_missing_exports"], [])
        # depends_on storage.py имеет exports → нет unmatched
        self.assertEqual(m["depends_unmatched"], [])
        self.assertEqual(m["depends_outside_plan"], [])

    def test_normalize_plan_contracts_flags_missing_exports_for_non_entry(self):
        plan = {
            "files": [
                {"path": "main.py", "depends_on": ["storage.py"], "exports": []},
                {"path": "storage.py", "depends_on": [], "exports": []},
            ]
        }
        out = self.pm._normalize_plan_contracts(plan, None)
        m = out["_contract_metrics"]
        # storage.py — не entry-point, exports пуст → попадает в missing
        self.assertIn("storage.py", m["files_missing_exports"])
        # main.py зависит от storage.py, у которого exports пуст → unmatched
        self.assertEqual(m["depends_unmatched"], ["main.py->storage.py"])

    def test_normalize_plan_contracts_depends_outside_plan(self):
        plan = {
            "files": [
                {"path": "main.py", "depends_on": ["missing_module.py"], "exports": []},
            ]
        }
        out = self.pm._normalize_plan_contracts(plan, None)
        self.assertEqual(
            out["_contract_metrics"]["depends_outside_plan"],
            ["main.py->missing_module.py"],
        )

    def test_normalize_plan_contracts_stdlib_not_outside(self):
        plan = {
            "files": [
                {"path": "main.py", "depends_on": ["stdlib", "STDLIB"], "exports": []},
            ]
        }
        out = self.pm._normalize_plan_contracts(plan, None)
        self.assertEqual(out["_contract_metrics"]["depends_outside_plan"], [])

    def test_normalize_plan_contracts_drops_garbage_exports(self):
        plan = {
            "files": [
                {"path": "util.py", "depends_on": [],
                 "exports": [
                     {"name": "good_fn"},
                     None,
                     {"name": "bad name"},
                     {"name": ""},
                     "not a dict",
                 ]},
            ]
        }
        out = self.pm._normalize_plan_contracts(plan, None)
        names = [e["name"] for e in out["files"][0]["exports"]]
        self.assertEqual(names, ["good_fn"])

    def test_normalize_plan_contracts_non_python_no_export_required(self):
        # data.json без exports — не учитывается в py_files и не считается missing
        plan = {
            "files": [
                {"path": "main.py", "depends_on": ["data.json"], "exports": []},
                {"path": "data.json", "depends_on": []},
            ]
        }
        out = self.pm._normalize_plan_contracts(plan, None)
        m = out["_contract_metrics"]
        self.assertEqual(m["py_files"], 1)
        # main.py зависит от data.json (не-питон) → не unmatched
        self.assertEqual(m["depends_unmatched"], [])


class TestCoderNeighborApiP11_2(unittest.TestCase):
    """P11.2: coder получает API соседей (read-only stubs + CONTRACT-блок + post-lint)."""

    @classmethod
    def setUpClass(cls):
        import brain.agents.project as project_mod
        cls.pm = project_mod

    # --- _module_name_from_rel ---

    def test_module_name_simple(self):
        self.assertEqual(self.pm._module_name_from_rel("main.py"), "main")

    def test_module_name_subdir(self):
        self.assertEqual(self.pm._module_name_from_rel("src/utils.py"), "src.utils")

    def test_module_name_backslash(self):
        self.assertEqual(self.pm._module_name_from_rel("src\\utils.py"), "src.utils")

    def test_module_name_empty(self):
        self.assertEqual(self.pm._module_name_from_rel(""), "")
        self.assertEqual(self.pm._module_name_from_rel(None), "")

    # --- _render_export_signature ---

    def test_render_signature_function_with_paren_sig(self):
        s = self.pm._render_export_signature({"name": "add", "kind": "function", "signature": "(a, b)"})
        self.assertEqual(s, "add(a, b)")

    def test_render_signature_function_with_full_sig(self):
        s = self.pm._render_export_signature({"name": "add", "kind": "function",
                                              "signature": "add(a: int, b: int) -> int"})
        self.assertEqual(s, "add(a: int, b: int) -> int")

    def test_render_signature_function_no_sig(self):
        s = self.pm._render_export_signature({"name": "main", "kind": "function"})
        self.assertEqual(s, "main()")

    def test_render_signature_class_with_init_sig(self):
        s = self.pm._render_export_signature({"name": "Storage", "kind": "class", "signature": "(db_path: str)"})
        self.assertEqual(s, "class Storage(db_path: str)")

    def test_render_signature_class_no_sig(self):
        s = self.pm._render_export_signature({"name": "User", "kind": "class"})
        self.assertEqual(s, "class User")

    def test_render_signature_const_with_type(self):
        s = self.pm._render_export_signature({"name": "DB_PATH", "kind": "const", "signature": "str"})
        self.assertEqual(s, "DB_PATH: str")

    def test_render_signature_const_bare(self):
        s = self.pm._render_export_signature({"name": "MAX", "kind": "const"})
        self.assertEqual(s, "MAX")

    def test_render_signature_garbage(self):
        self.assertEqual(self.pm._render_export_signature(None), "")
        self.assertEqual(self.pm._render_export_signature({}), "")

    # --- _render_neighbor_stub ---

    def test_neighbor_stub_function_is_valid_python(self):
        import ast
        stub = self.pm._render_neighbor_stub("storage.py", {
            "purpose": "Работа с SQLite",
            "exports": [
                {"name": "add_user", "kind": "function", "signature": "(name: str) -> int", "doc": "Добавить."},
            ],
        })
        # Должен парситься как Python
        ast.parse(stub)
        self.assertIn("def add_user", stub)
        self.assertIn("NotImplementedError", stub)

    def test_neighbor_stub_class_is_valid_python(self):
        import ast
        stub = self.pm._render_neighbor_stub("storage.py", {
            "exports": [
                {"name": "Storage", "kind": "class", "signature": "(db_path: str)"},
            ],
        })
        ast.parse(stub)  # обязан быть валидным Python
        self.assertIn("class Storage:", stub)
        # Сигнатуру показываем в комментарии
        self.assertIn("db_path: str", stub)

    def test_neighbor_stub_const_is_valid_python(self):
        import ast
        stub = self.pm._render_neighbor_stub("config.py", {
            "exports": [
                {"name": "DB_PATH", "kind": "const", "signature": "str"},
            ],
        })
        ast.parse(stub)
        self.assertIn("DB_PATH = None", stub)

    def test_neighbor_stub_empty_exports(self):
        import ast
        stub = self.pm._render_neighbor_stub("empty.py", {"exports": []})
        ast.parse(stub)  # всё равно валидный Python

    # --- _build_neighbor_context ---

    def test_build_neighbor_context_uses_real_file_when_built(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "storage.py").write_text("def real_fn(): return 1\n", encoding="utf-8")
            plan = {
                "files": [
                    {"path": "main.py", "depends_on": ["storage.py"], "exports": []},
                    {"path": "storage.py", "depends_on": [],
                     "exports": [{"name": "real_fn", "kind": "function"}]},
                ]
            }
            paths, descs = self.pm._build_neighbor_context(root, plan, "main.py")
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("storage.py"))
            self.assertTrue(any("экспортирует" in d for d in descs))

    def test_build_neighbor_context_generates_stub_when_not_built(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = {
                "files": [
                    {"path": "main.py", "depends_on": ["storage.py"], "exports": []},
                    {"path": "storage.py", "depends_on": [],
                     "exports": [{"name": "add_user", "kind": "function"}]},
                ]
            }
            paths, descs = self.pm._build_neighbor_context(root, plan, "main.py")
            self.assertEqual(len(paths), 1)
            stub_path = Path(paths[0])
            self.assertTrue(stub_path.exists())
            self.assertIn("def add_user", stub_path.read_text(encoding="utf-8"))

    def test_build_neighbor_context_skips_stdlib(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            plan = {
                "files": [
                    {"path": "main.py", "depends_on": ["stdlib"], "exports": []},
                ]
            }
            paths, descs = self.pm._build_neighbor_context(Path(td), plan, "main.py")
            self.assertEqual(paths, [])
            self.assertEqual(descs, [])

    def test_build_neighbor_context_skips_unknown_dep(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            plan = {
                "files": [
                    {"path": "main.py", "depends_on": ["missing_module.py"], "exports": []},
                ]
            }
            paths, descs = self.pm._build_neighbor_context(Path(td), plan, "main.py")
            self.assertEqual(paths, [])

    def test_build_neighbor_context_garbage_plan(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            paths, descs = self.pm._build_neighbor_context(Path(td), None, "main.py")
            self.assertEqual(paths, [])
            self.assertEqual(descs, [])

    # --- _build_contract_prompt_block ---

    def test_contract_block_must_export_section(self):
        plan = {"files": [
            {"path": "storage.py", "depends_on": ["stdlib"],
             "exports": [{"name": "add_user", "kind": "function", "signature": "(name: str)"}]}
        ]}
        block = self.pm._build_contract_prompt_block(plan, "storage.py")
        self.assertIn("КОНТРАКТ ЭТОГО ФАЙЛА", block)
        self.assertIn("add_user(name: str)", block)

    def test_contract_block_required_imports_section(self):
        plan = {"files": [
            {"path": "main.py", "depends_on": ["storage.py", "stdlib"], "exports": []},
            {"path": "storage.py", "depends_on": [],
             "exports": [{"name": "add_user", "kind": "function"},
                         {"name": "list_users", "kind": "function"}]},
        ]}
        block = self.pm._build_contract_prompt_block(plan, "main.py")
        self.assertIn("ОБЯЗАТЕЛЬНЫЕ ИМПОРТЫ", block)
        self.assertIn("from storage import add_user, list_users", block)

    def test_contract_block_empty_when_no_exports(self):
        plan = {"files": [
            {"path": "main.py", "depends_on": ["stdlib"], "exports": []},
        ]}
        block = self.pm._build_contract_prompt_block(plan, "main.py")
        self.assertEqual(block, "")

    def test_contract_block_garbage_plan(self):
        self.assertEqual(self.pm._build_contract_prompt_block(None, "main.py"), "")
        self.assertEqual(self.pm._build_contract_prompt_block({}, "main.py"), "")

    # --- _check_file_contract ---

    def test_check_file_contract_ok(self):
        text = "def add_user(name): pass\nDB_PATH = 'x'\n"
        cc = self.pm._check_file_contract(text, [
            {"name": "add_user", "kind": "function"},
            {"name": "DB_PATH", "kind": "const"},
        ])
        self.assertTrue(cc["ok"])
        self.assertEqual(cc["missing"], [])
        self.assertEqual(cc["kind_mismatch"], [])

    def test_check_file_contract_missing(self):
        text = "def add_user(name): pass\n"
        cc = self.pm._check_file_contract(text, [
            {"name": "add_user", "kind": "function"},
            {"name": "DB_PATH", "kind": "const"},
        ])
        self.assertFalse(cc["ok"])
        self.assertIn("DB_PATH", cc["missing"])

    def test_check_file_contract_kind_mismatch(self):
        text = "def Storage(): pass\n"
        cc = self.pm._check_file_contract(text, [
            {"name": "Storage", "kind": "class"},
        ])
        self.assertFalse(cc["ok"])
        self.assertEqual(cc["kind_mismatch"], [
            {"name": "Storage", "expected": "class", "actual": "function"}
        ])

    def test_check_file_contract_renamed_const(self):
        # Реальный сценарий из reminder-bot: plan ждёт DB_PATH, coder выдал DATABASE_PATH.
        text = "DATABASE_PATH = 'reminders.sqlite'\n"
        cc = self.pm._check_file_contract(text, [
            {"name": "DB_PATH", "kind": "const"},
        ])
        self.assertFalse(cc["ok"])
        self.assertEqual(cc["missing"], ["DB_PATH"])
        self.assertIn("DATABASE_PATH", cc["found_top_level"])

    def test_check_file_contract_syntax_error(self):
        cc = self.pm._check_file_contract("def x(:", [{"name": "x", "kind": "function"}])
        self.assertFalse(cc["ok"])
        self.assertFalse(cc["ast_ok"])

    def test_check_file_contract_empty_text(self):
        cc = self.pm._check_file_contract("", [{"name": "foo", "kind": "function"}])
        self.assertFalse(cc["ok"])
        self.assertEqual(cc["missing"], ["foo"])

    def test_check_file_contract_no_expected(self):
        cc = self.pm._check_file_contract("x = 1\n", [])
        self.assertTrue(cc["ok"])

    # --- _dedupe_files_vs_inputs (FM-10) ---

    def test_dedupe_removes_input_from_files(self):
        plan = {
            "files": [
                {"path": "storage.py", "depends_on": []},
                {"path": "todos.json", "depends_on": []},
            ],
            "inputs": [{"path": "todos.json", "sample_content": "[]"}],
        }
        out = self.pm._dedupe_files_vs_inputs(plan)
        paths = [f["path"] for f in out["files"]]
        self.assertEqual(paths, ["storage.py"])

    def test_dedupe_keeps_python_overlap(self):
        # Если в inputs оживает .py-файл (баг плана) — не режем, пусть строится.
        plan = {
            "files": [{"path": "main.py"}],
            "inputs": [{"path": "main.py"}],
        }
        out = self.pm._dedupe_files_vs_inputs(plan)
        self.assertEqual([f["path"] for f in out["files"]], ["main.py"])

    def test_dedupe_empty_inputs_no_change(self):
        plan = {"files": [{"path": "main.py"}], "inputs": []}
        out = self.pm._dedupe_files_vs_inputs(plan)
        self.assertEqual([f["path"] for f in out["files"]], ["main.py"])

    def test_dedupe_garbage_plan(self):
        self.assertEqual(self.pm._dedupe_files_vs_inputs(None), None)
        out = self.pm._dedupe_files_vs_inputs({})
        self.assertIsInstance(out, dict)

    # --- aider_runner _build_argv read-only ---

    def test_aider_argv_includes_read_flag(self):
        from pathlib import Path
        from brain.agents import aider_runner
        argv = aider_runner._build_argv(
            Path("/tmp/x"), "main.py", "do thing",
            model="ollama/x", api_base="http://localhost:11434",
            read_only_files=["/abs/storage.py", "/abs/config.py"],
        )
        # Должны быть две пары --read <path>
        self.assertEqual(argv.count("--read"), 2)
        idx1 = argv.index("--read")
        self.assertEqual(argv[idx1 + 1], "/abs/storage.py")

    def test_aider_argv_no_read_when_empty(self):
        from pathlib import Path
        from brain.agents import aider_runner
        argv = aider_runner._build_argv(
            Path("/tmp/x"), "main.py", "do thing",
            model="ollama/x", api_base="http://localhost:11434",
        )
        self.assertNotIn("--read", argv)


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



class TestCrossFileHealP11_4(unittest.TestCase):
    """P11.4: структурный разбор ошибки + выбор правильного target.

    Тестируем _classify_failure на реальных traceback-форматах, без LLM.
    Все решения должны идти из формата ошибки + plan.exports — без угадывания по словам.
    """

    # ─── _classify_failure: ImportError ──────────────────────────────────

    def test_import_error_picks_owner_file(self):
        """ImportError 'X' from M → target = файл, где X объявлен в plan.exports."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "config.py", "exports": [{"name": "DB_PATH", "kind": "const"}]},
            {"path": "main.py",   "exports": []},
        ]}
        failed = {"stderr": (
            'Traceback (most recent call last):\n'
            '  File "C:\\j\\main.py", line 5, in <module>\n'
            '    from config import DB_PATH\n'
            "ImportError: cannot import name 'DB_PATH' from 'config'\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["kind"], "import_error")
        self.assertEqual(info["target"], "config.py")
        self.assertEqual(info["missing_name"], "DB_PATH")
        self.assertIn("DB_PATH", info["hint"])

    def test_import_error_unknown_name_falls_to_module(self):
        """ImportError 'X' from M, но X нет в exports — target = файл модуля M."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "config.py", "exports": []},
            {"path": "main.py",   "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 1, in <module>\n'
            '    from config import UNKNOWN\n'
            "ImportError: cannot import name 'UNKNOWN' from 'config'\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["kind"], "import_error")
        self.assertEqual(info["target"], "config.py")

    def test_import_error_unresolved_module_uses_source_frame(self):
        """ImportError 'X' from M, M не в плане — target = source_file."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "main.py", "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 1, in <module>\n'
            '    from external_lib import Thing\n'
            "ImportError: cannot import name 'Thing' from 'external_lib'\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["kind"], "import_error")
        # owner нет, mod_path нет — fallback на source_file
        self.assertEqual(info["target"], "main.py")

    # ─── _classify_failure: AttributeError ──────────────────────────────

    def test_attribute_error_module_missing_attr(self):
        """AttributeError: module M has no attribute X → target = файл X (если X в plan.exports)."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "storage.py", "exports": [{"name": "Storage", "kind": "class"}]},
            {"path": "main.py",    "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 5, in <module>\n'
            '    s = storage.Storage()\n'
            "AttributeError: module 'storage' has no attribute 'Storage'\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["kind"], "attribute_error")
        self.assertEqual(info["target"], "storage.py")
        self.assertEqual(info["missing_name"], "Storage")

    def test_attribute_error_unknown_owner_falls_to_module(self):
        """AttributeError, имя нет в exports — target = файл модуля."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "storage.py", "exports": []},
            {"path": "main.py",    "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 5, in <module>\n'
            '    storage.unknown()\n'
            "AttributeError: module 'storage' has no attribute 'unknown'\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["target"], "storage.py")

    # ─── _classify_failure: NameError ────────────────────────────────────

    def test_name_error_local_var_targets_source_file(self):
        """NameError 'args' в main.py → target = main.py (баг там, где использовалось)."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "main.py", "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 28, in <module>\n'
            '    elif args.command == "list":\n'
            "NameError: name 'args' is not defined\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["kind"], "name_error")
        self.assertEqual(info["target"], "main.py")
        self.assertEqual(info["missing_name"], "args")

    def test_name_error_neighbor_export_suggests_import(self):
        """NameError 'Storage' в main.py, Storage — экспорт соседа → target = main.py + hint про import."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "storage.py", "exports": [{"name": "Storage", "kind": "class"}]},
            {"path": "main.py",    "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 3, in <module>\n'
            '    s = Storage()\n'
            "NameError: name 'Storage' is not defined\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["target"], "main.py")
        self.assertIn("from storage import", info["hint"])
        self.assertIn("Storage", info["hint"])

    def test_name_error_owner_same_as_source_uses_local_path(self):
        """NameError 'X', X в exports того же source_file — берём локальную ветку (баг там же)."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "storage.py", "exports": [{"name": "Storage", "kind": "class"}]},
        ]}
        failed = {"stderr": (
            'File "storage.py", line 5, in <module>\n'
            '    Storage()\n'
            "NameError: name 'Storage' is not defined\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["target"], "storage.py")

    # ─── _last_user_frame_in_plan ────────────────────────────────────────

    def test_last_user_frame_picks_last_in_plan(self):
        """Из traceback с несколькими user-frames берём последний (где упало)."""
        from brain.agents.project import _last_user_frame_in_plan
        stderr = (
            'File "main.py", line 1, in <module>\n'
            '    bot.run()\n'
            'File "bot.py", line 10, in run\n'
            '    storage.add()\n'
            'File "storage.py", line 5, in add\n'
            "    raise ValueError('boom')\n"
        )
        result = _last_user_frame_in_plan(stderr, ["main.py", "bot.py", "storage.py"])
        self.assertEqual(result, "storage.py")

    def test_last_user_frame_skips_stdlib(self):
        """Frames вне plan.files игнорируются (stdlib, библиотеки)."""
        from brain.agents.project import _last_user_frame_in_plan
        stderr = (
            'File "main.py", line 1, in <module>\n'
            '    json.loads(s)\n'
            'File "C:\\Python\\Lib\\json\\__init__.py", line 357, in loads\n'
            '    return _default_decoder.decode(s)\n'
        )
        result = _last_user_frame_in_plan(stderr, ["main.py"])
        # последний frame вне плана — берём предыдущий из плана
        self.assertEqual(result, "main.py")

    def test_last_user_frame_handles_windows_paths(self):
        """Windows backslash-пути из реального запуска нормально матчатся."""
        from brain.agents.project import _last_user_frame_in_plan
        stderr = 'File "C:\\jarvis\\data\\projects\\xyz\\bot.py", line 10, in run\n'
        result = _last_user_frame_in_plan(stderr, ["bot.py"])
        self.assertEqual(result, "bot.py")

    def test_last_user_frame_none_when_no_match(self):
        from brain.agents.project import _last_user_frame_in_plan
        stderr = 'File "external.py", line 1, in <module>\n'
        result = _last_user_frame_in_plan(stderr, ["main.py"])
        self.assertIsNone(result)

    # ─── _module_to_plan_path ────────────────────────────────────────────

    def test_module_to_plan_path_simple(self):
        from brain.agents.project import _module_to_plan_path
        self.assertEqual(_module_to_plan_path("config", ["config.py", "main.py"]), "config.py")

    def test_module_to_plan_path_nested(self):
        from brain.agents.project import _module_to_plan_path
        self.assertEqual(_module_to_plan_path("src.utils", ["src/utils.py", "main.py"]), "src/utils.py")

    def test_module_to_plan_path_unknown(self):
        from brain.agents.project import _module_to_plan_path
        self.assertIsNone(_module_to_plan_path("unknown_pkg", ["main.py"]))

    # ─── _pick_heal_target — интеграция ──────────────────────────────────

    def test_pick_heal_target_uses_classification(self):
        """Если _classify_failure нашёл target — используем его, а не первый файл."""
        from brain.agents.project import _pick_heal_target
        plan = {"files": [
            {"path": "main.py",    "exports": []},
            {"path": "config.py",  "exports": [{"name": "DB_PATH", "kind": "const"}]},
        ]}
        failed = {"stderr": (
            'File "main.py", line 1, in <module>\n'
            "    from config import DB_PATH\n"
            "ImportError: cannot import name 'DB_PATH' from 'config'\n"
        )}
        # main.py — первый файл, но classify_failure должен вернуть config.py
        target = _pick_heal_target(plan, failed)
        self.assertEqual(target, "config.py")

    def test_pick_heal_target_fallback_when_no_clue(self):
        """Если ошибка не разобрана — fallback на первый .py."""
        from brain.agents.project import _pick_heal_target
        plan = {"files": [
            {"path": "main.py",    "exports": []},
            {"path": "config.py",  "exports": []},
        ]}
        failed = {"stderr": "some opaque error without traceback"}
        target = _pick_heal_target(plan, failed)
        self.assertEqual(target, "main.py")

    def test_pick_heal_target_no_files_returns_none(self):
        from brain.agents.project import _pick_heal_target
        self.assertIsNone(_pick_heal_target({"files": []}, {}))

    # ─── Регрессия P11.2: reminder-bot DB_PATH должен попасть в config.py ─

    def test_regression_reminder_bot_db_path_targets_config(self):
        """Реальный кейс P11.2: heal должен выбрать config.py, а не main.py.

        До P11.4: _pick_heal_target выбирал config.py, потому что 'config.py' в stderr —
        случайное совпадение. С P11.4: тот же выбор, но обоснованно через plan.exports.
        А для NameError 'args' выбирается main.py (раньше тоже мог пойти не туда).
        """
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "config.py",  "exports": [{"name": "DB_PATH", "kind": "const"}]},
            {"path": "storage.py", "exports": [
                {"name": "Storage", "kind": "class"},
                {"name": "add_reminder", "kind": "function"},
                {"name": "list_reminders", "kind": "function"},
            ]},
            {"path": "bot.py",     "exports": [{"name": "process_command", "kind": "function"}]},
            {"path": "main.py",    "exports": []},
        ]}
        # ImportError: должен идти в config.py
        f1 = {"stderr": (
            'File "main.py", line 5, in <module>\n'
            '    from config import DB_PATH\n'
            "ImportError: cannot import name 'DB_PATH' from 'config'\n"
        )}
        self.assertEqual(_classify_failure(plan, f1)["target"], "config.py")

        # NameError на Storage (не импортирован в bot.py): должен идти в bot.py + hint про import
        f2 = {"stderr": (
            'File "bot.py", line 10, in process_command\n'
            '    s = Storage()\n'
            "NameError: name 'Storage' is not defined\n"
        )}
        info = _classify_failure(plan, f2)
        self.assertEqual(info["target"], "bot.py")
        self.assertIn("from storage import", info["hint"])

    # ─── Регрессия P11.2: NameError args в main.py ───────────────────────

    def test_regression_todo_args_undefined_targets_main(self):
        """Реальный кейс P11.2: NameError 'args' в main.py."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "storage.py", "exports": [{"name": "load_todos"}, {"name": "save_todos"}]},
            {"path": "cli.py",     "exports": [{"name": "add_todo"}, {"name": "list_todos"}]},
            {"path": "main.py",    "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 28, in <module>\n'
            '    elif args.command == "list":\n'
            "NameError: name 'args' is not defined\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["target"], "main.py")
        self.assertEqual(info["missing_name"], "args")
        # 'args' не в exports никого — просто локально не определена
        self.assertNotIn("from", info["hint"].split("\n")[0])

    # ─── Защита: пустой/ломаный input ────────────────────────────────────

    def test_classify_empty_plan(self):
        from brain.agents.project import _classify_failure
        info = _classify_failure({"files": []}, {"stderr": "anything"})
        self.assertEqual(info["kind"], "unknown")
        self.assertIsNone(info["target"])

    def test_classify_empty_stderr(self):
        from brain.agents.project import _classify_failure
        plan = {"files": [{"path": "main.py", "exports": []}]}
        info = _classify_failure(plan, {"stderr": "", "stdout": ""})
        self.assertIsNone(info["target"])

    def test_classify_non_dict_args(self):
        from brain.agents.project import _classify_failure
        # не падать
        info = _classify_failure(None, None)
        self.assertEqual(info["kind"], "unknown")
        info2 = _classify_failure([], "stderr")
        self.assertEqual(info2["kind"], "unknown")

    # ─── traceback без специфической ошибки ──────────────────────────────

    def test_generic_traceback_targets_last_frame(self):
        """ValueError или другая ошибка с user-frame — target = последний frame."""
        from brain.agents.project import _classify_failure
        plan = {"files": [
            {"path": "main.py",    "exports": []},
            {"path": "storage.py", "exports": []},
        ]}
        failed = {"stderr": (
            'File "main.py", line 1, in <module>\n'
            '    save()\n'
            'File "storage.py", line 5, in save\n'
            "    raise ValueError('disk full')\n"
            "ValueError: disk full\n"
        )}
        info = _classify_failure(plan, failed)
        self.assertEqual(info["kind"], "traceback")
        self.assertEqual(info["target"], "storage.py")

    # ─── Quick-win A: detail обрезка в jsonl увеличена ───────────────────

    def test_phase_detail_jsonl_limit_is_large(self):
        """detail в jsonl должен теперь сохранять минимум 4000 символов (P11.4)."""
        from tools.projects import _PHASE_DETAIL_JSONL_LIMIT
        self.assertGreaterEqual(_PHASE_DETAIL_JSONL_LIMIT, 4000)
