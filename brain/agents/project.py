"""
brain/agents/project.py — ProjectAgent (Level 4: «Создатель»).

Полный цикл от запроса пользователя до работающего проекта с тестами,
изолированным venv, README и записью в _index.jsonl.

Фазы (каждая логируется в manifest.json и logs/projects.jsonl):
  1. INTAKE     — извлекает спеку из NL-запроса (PROJECT_INTAKE_SYSTEM, fast)
  2. ARCHITECT  — проектирует файлы и тесты (PROJECT_ARCHITECT_SYSTEM, heavy)
  3. ENV        — создаёт venv и ставит install_dep пакеты из плана
  4. BUILD      — для каждого файла write→review→fix цикл (max MAX_REVIEW_ITERS).
                  Каждый Coder получает уже-написанные файлы как контекст.
  5. TEST       — запускает test commands внутри venv, проверяет expects в stdout
  6. HEAL       — если тест упал, Healer диагностирует → Coder патчит → перетест
                  (до MAX_HEAL_ITERS). Цикл прерывается ранним успехом.
  7. README     — генерирует README.md в корне проекта
  8. REPORT     — короткий устный итог + запись в _index.jsonl

Бюджеты (защита от бесконечности):
  - PROJECT_WALL_BUDGET_S — общий wall-clock на проект (default 600)
  - PROJECT_LLM_BUDGET    — общий лимит LLM-вызовов (default 40)
  При превышении — фаза Pomeguard прерывает текущий шаг, статус='failed',
  но manifest сохраняется и --resume работает.

Resume:
  brain.agents.project.resume(slug) — продолжает упавший проект с упавшей фазы.
  Использует last_phase из манифеста.

CLI:
  python -m brain.agents.project "<запрос>"
  python -m brain.agents.project --resume <slug>
  python -m brain.agents.project --list
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from typing import Any

from brain.client import chat, MODEL_FAST, MODEL_HEAVY
from brain.prompts import (
    PROJECT_INTAKE_SYSTEM,
    PROJECT_ARCHITECT_SYSTEM,
    PROJECT_REPORT_SYSTEM,
    PROJECT_HEAL_SYSTEM,
    PROJECT_README_SYSTEM,
)
from brain.agents import coder as coder_agent
from brain.agents import reviewer as reviewer_agent
from tools.static_checks import (
    static_check,
    static_errors_to_feedback,
    static_warnings_to_hint,
)
from tools.projects import (
    create_project,
    write_project_file,
    read_project_file,
    add_phase,
    set_status,
    save_manifest,
    load_manifest,
    list_projects,
    run_in_project,
    run_with_project_python,
    python_smoke,
    ensure_venv,
    pip_install,
    append_index_record,
    get_project_files,
    project_dir,
    safe_project_path,
)

logger = logging.getLogger(__name__)

MAX_REVIEW_ITERS = 2
MAX_HEAL_ITERS   = 2
MAX_FILES        = 10
PHASE_TEST_TIMEOUT = 30
PROJECT_WALL_BUDGET_S = 600       # 10 минут на проект целиком
PROJECT_LLM_BUDGET    = 40        # суммарно на все фазы


# ─── budget tracking ────────────────────────────────────────────────────────
class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, wall_s: float = PROJECT_WALL_BUDGET_S, llm: int = PROJECT_LLM_BUDGET):
        self.start  = time.monotonic()
        self.wall_s = wall_s
        self.llm_max = llm
        self.llm_used = 0

    def check(self, where: str = "") -> None:
        if time.monotonic() - self.start > self.wall_s:
            raise BudgetExceeded(f"wall-clock budget exhausted at {where}")
        if self.llm_used >= self.llm_max:
            raise BudgetExceeded(f"llm-call budget exhausted at {where}")

    def spend(self, n: int = 1) -> None:
        self.llm_used += n

    def remaining_s(self) -> float:
        return max(0.0, self.wall_s - (time.monotonic() - self.start))

    def summary(self) -> dict:
        return {
            "wall_s_used":  round(time.monotonic() - self.start, 2),
            "wall_s_max":   self.wall_s,
            "llm_used":     self.llm_used,
            "llm_max":      self.llm_max,
        }


# ─── helpers ────────────────────────────────────────────────────────────────
def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _safe_parse(raw: str) -> dict:
    txt = _strip_json_fence(raw)
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _llm(budget: Budget, model: str, system: str, user: str, *,
         temperature: float = 0.1, num_ctx: int = 4096, where: str = "") -> str:
    budget.check(where)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = chat(model, msgs, options={"temperature": temperature, "num_ctx": num_ctx})
    budget.spend(1)
    return out


def _set_last_phase(slug: str, name: str) -> None:
    try:
        m = load_manifest(slug)
        m.last_phase = name
        save_manifest(m)
    except Exception:
        pass


def _save_metrics(slug: str, **fields) -> None:
    try:
        m = load_manifest(slug)
        m.metrics.update(fields)
        save_manifest(m)
    except Exception:
        pass


# ─── PHASE 1: intake ────────────────────────────────────────────────────────
def _intake(query: str, budget: Budget) -> dict:
    raw = _llm(budget, MODEL_FAST, PROJECT_INTAKE_SYSTEM, query,
               temperature=0.1, num_ctx=4096, where="intake")
    spec = _safe_parse(raw)
    if not isinstance(spec, dict) or not spec.get("title"):
        spec = {
            "title": "untitled-project",
            "slug": "",
            "kind": "script",
            "language": "python",
            "summary": query[:200],
            "requirements": [query[:200]],
            "deliverables": ["main.py"],
            "acceptance_criteria": ["скрипт запускается без ошибок"],
        }
    spec.setdefault("requirements", [])
    spec.setdefault("deliverables", [])
    spec.setdefault("acceptance_criteria", [])
    return spec


# ─── PHASE 2: architect ─────────────────────────────────────────────────────
def _architect(spec: dict, budget: Budget) -> dict:
    user = "Спецификация проекта:\n" + json.dumps(spec, ensure_ascii=False, indent=2)
    raw = _llm(budget, MODEL_HEAVY, PROJECT_ARCHITECT_SYSTEM, user,
               temperature=0.1, num_ctx=8192, where="architect")
    plan = _safe_parse(raw)
    files = plan.get("files") or []
    if not isinstance(files, list) or not files:
        plan = {
            "files": [{"path": "main.py", "purpose": "точка входа", "depends_on": ["stdlib"]}],
            "build_steps": [
                {"step": 1, "kind": "create_file", "target": "main.py", "description": "создаём entry"},
                {"step": 2, "kind": "smoke_run",   "target": "python main.py", "description": "smoke"},
            ],
            "tests": [{"name": "smoke", "command": "python main.py", "expects": ""}],
        }
    plan["files"] = (plan.get("files") or [])[:MAX_FILES]
    plan.setdefault("build_steps", [])
    plan.setdefault("tests", [])
    return plan


# ─── PHASE 3: env (venv + pip) ──────────────────────────────────────────────
_PKG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]+([<>=!~]=?[A-Za-z0-9._-]+)?$")


# Стоп-слова — это НЕ пакеты, а флаги/опции/команды
_PKG_STOPWORDS = {
    "pip", "install", "-r", "--requirement", "-U", "--upgrade", "--user",
    "--no-input", "--no-deps", "--prefer-binary", "requirements.txt",
    "-e", "--editable", ".", "./", "venv", "python", "-m",
}


def _parse_requirements_txt(content: str) -> list[str]:
    """Разобрать requirements.txt — по одному пакету на строку, игнорируя # комментарии."""
    pkgs: list[str] = []
    for raw in (content or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Пропускаем строки вида -r other.txt, -e ., и прочую флаговую муть
        if line.startswith("-") or line in _PKG_STOPWORDS:
            continue
        if _PKG_PATTERN.match(line):
            pkgs.append(line)
    return pkgs


def _extract_packages(slug: str, plan: dict) -> list[str]:
    """Собирает пакеты из трёх источников (по убыванию надёжности):
      1. plan['pip_requirements'] — явно объявленные архитектором.
      2. Содержимое requirements.txt в проекте (если файл уже сгенерирован).
      3. build_steps[install_dep].target — легаси формат, фильтруем стоп-слова.
    """
    pkgs: list[str] = []

    # Источник 1: явный pip_requirements
    for p in (plan.get("pip_requirements") or []):
        if isinstance(p, str):
            tok = p.strip().strip("\"'")
            if tok and _PKG_PATTERN.match(tok) and tok.lower() not in _PKG_STOPWORDS:
                pkgs.append(tok)

    # Источник 2: requirements.txt в файлах проекта
    try:
        existing = get_project_files(slug)
        if isinstance(existing, dict) and "requirements.txt" in existing:
            pkgs.extend(_parse_requirements_txt(existing["requirements.txt"]))
    except Exception as e:
        logger.debug(f"[project] requirements.txt parse skipped: {e}")

    # Источник 3: build_steps[install_dep] — legacy fallback
    for step in (plan.get("build_steps") or []):
        if step.get("kind") != "install_dep":
            continue
        target = step.get("target") or ""
        for token in re.split(r"[,\s]+", target):
            token = token.strip().strip("\"'")
            if not token:
                continue
            if token.lower() in _PKG_STOPWORDS:
                continue
            # Пропускаем явные пути к файлам (содержат / или \ или заканчиваются на .txt)
            if "/" in token or "\\" in token or token.endswith(".txt"):
                continue
            if _PKG_PATTERN.match(token):
                pkgs.append(token)

    # Дедуп с сохранением порядка (case-insensitive)
    seen: set[str] = set()
    out: list[str] = []
    for p in pkgs:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _phase_env(slug: str, plan: dict) -> dict:
    venv_res = ensure_venv(slug)
    if not venv_res["ok"]:
        return {"ok": False, "error": f"venv: {venv_res.get('error')}"}
    pkgs = _extract_packages(slug, plan)
    if not pkgs:
        return {"ok": True, "venv": True, "installed": [], "skipped_install": True}
    install_res = pip_install(slug, pkgs)
    return {
        "ok": install_res.get("ok", False),
        "venv": True,
        "installed": install_res.get("installed", []),
        "requested": pkgs,
        "stderr": install_res.get("stderr", ""),
    }


# ─── PHASE 4: build (coder ↔ reviewer loop) ────────────────────────────────
def _build_one_file(slug: str, spec: dict, plan: dict, target: dict, budget: Budget) -> dict:
    """Build-loop с детерминистической статикой перед LLM-Reviewer (P1).

    На каждой итерации:
      1. Coder пишет/патчит код.
      2. static_check (ast.parse + ruff/pyflakes если есть).
      3. Если синтаксис битый → пропускаем LLM-Reviewer, ast-ошибка — feedback.
      4. Иначе вызываем Reviewer (при наличии lint-warnings — пробросим их как hint).
    """
    feedback = ""
    code = ""
    final_review: dict[str, Any] = {}
    last_static: dict[str, Any] = {}
    existing = get_project_files(slug)
    static_fail_streak = 0
    for it in range(MAX_REVIEW_ITERS + 1):
        budget.check(f"build:{target.get('path')}:iter{it}")
        if it == 0:
            code = coder_agent.write_file(spec, plan, target, existing=existing)
        else:
            code = coder_agent.patch_file(spec, plan, target, code, feedback, existing=existing)
        budget.spend(1)

        # P1: детерминистическая статика до Reviewer.
        sc = static_check(target.get("path", ""), code)
        last_static = sc
        if sc.get("applicable") and not sc.get("ok"):
            # Синтаксис битый — не тратим LLM на Reviewer.
            static_fail_streak += 1
            feedback = static_errors_to_feedback(sc.get("errors") or [])
            final_review = {
                "verdict": "revise",
                "issues": [{"severity": "error", "message": e} for e in sc.get("errors", [])],
                "summary": f"static: {sc.get('errors', [''])[0][:120]}",
                "_source": "static",
            }
            # Если это последняя допустимая итерация — выходим, файл останется писаться как
            # есть (пусть хилер ловит на фазе test или проект упадёт честно).
            if it >= MAX_REVIEW_ITERS:
                break
            continue

        budget.check(f"review:{target.get('path')}:iter{it}")
        # Пробрасываем lint-warnings как hint Reviewer'у через spec'овый слот без ломки API.
        # Reviewer.review не знает про это, но в promt-сборке spec выводится целиком — линт будет виден.
        review_spec = spec
        warnings = sc.get("warnings") or []
        if warnings:
            review_spec = {**spec, "_static_lint_hint": static_warnings_to_hint(warnings)}
        rv = reviewer_agent.review(review_spec, target, code)
        budget.spend(1 if rv.get("_source") == "llm" else 0)
        final_review = rv
        if rv["verdict"] == "approve":
            break
        feedback = reviewer_agent.issues_as_feedback(rv["issues"])
        if not feedback:
            break

    try:
        write_project_file(slug, target["path"], code)
    except Exception as e:
        logger.error(f"[project] write_project_file failed: {e}")
        return {"path": target.get("path"), "ok": False, "error": str(e)}

    return {
        "path":     target["path"],
        "ok":       True,
        "verdict":  final_review.get("verdict"),
        "issues":   len(final_review.get("issues") or []),
        "summary":  final_review.get("summary", "")[:200],
        "iters":    it + 1,
        "static":   {
            "tools":          last_static.get("tools") or [],
            "errors":         last_static.get("errors") or [],
            "warnings":       last_static.get("warnings") or [],
            "fail_streak":    static_fail_streak,
            "final_ast_ok":   bool(last_static.get("ok", True)),
        },
    }


def _build(slug: str, spec: dict, plan: dict, budget: Budget) -> list[dict]:
    results = []
    for target in plan["files"]:
        if not isinstance(target, dict) or "path" not in target:
            continue
        try:
            res = _build_one_file(slug, spec, plan, target, budget)
        except BudgetExceeded as e:
            res = {"path": target.get("path"), "ok": False, "error": f"budget: {e}"}
            results.append(res)
            add_phase(slug, f"build:{target.get('path')}", "failed", str(e))
            break
        results.append(res)
        status = "ok" if res.get("ok") else "failed"
        add_phase(slug, f"build:{target.get('path')}", status, json.dumps(res, ensure_ascii=False))
    return results


# ─── PHASE 5: test ──────────────────────────────────────────────────────────
# Машинно-проверяемые типы checks. Заменяют свободнотекстовый `expects`,
# который заставлял Coder хардкодить ожидаемые строки в код.
VALID_CHECK_TYPES = {
    "rc_zero",          # код возврата == 0
    "file_exists",      # path существует
    "file_min_size",    # path имеет размер >= bytes
    "file_min_lines",   # path имеет >= lines строк
    "stdout_contains",  # text встречается в stdout (case-insensitive)
}


def _evaluate_check(slug: str, check: dict, run_result: dict) -> dict:
    """Проверяет одно условие против результата запуска. Возвращает dict с ok/reason."""
    if not isinstance(check, dict):
        return {"type": "invalid", "ok": False, "reason": "check is not a dict"}
    ctype = (check.get("type") or "").strip()
    if ctype not in VALID_CHECK_TYPES:
        return {"type": ctype, "ok": False, "reason": f"unknown check type: {ctype!r}"}

    if ctype == "rc_zero":
        rc = run_result.get("returncode")
        return {"type": ctype, "ok": rc == 0, "reason": f"rc={rc}"}

    if ctype == "stdout_contains":
        text = (check.get("text") or "").strip()
        if not text:
            return {"type": ctype, "ok": False, "reason": "empty text"}
        stdout = (run_result.get("stdout") or "")
        ok = text.lower() in stdout.lower()
        return {"type": ctype, "ok": ok, "reason": f"text={text!r} found={ok}"}

    # Файловые проверки: путь относительно корня проекта, защищаем safe_project_path
    rel = (check.get("path") or "").strip()
    if not rel:
        return {"type": ctype, "ok": False, "reason": "empty path"}
    try:
        abs_path = safe_project_path(slug, rel)
    except Exception as e:
        return {"type": ctype, "ok": False, "reason": f"unsafe path: {e}"}

    if ctype == "file_exists":
        ok = abs_path.exists() and abs_path.is_file()
        return {"type": ctype, "ok": ok, "reason": f"path={rel} exists={ok}"}

    if ctype == "file_min_size":
        try:
            min_bytes = int(check.get("bytes", 0))
        except (TypeError, ValueError):
            return {"type": ctype, "ok": False, "reason": "bad bytes value"}
        if not abs_path.exists():
            return {"type": ctype, "ok": False, "reason": f"file not found: {rel}"}
        actual = abs_path.stat().st_size
        return {"type": ctype, "ok": actual >= min_bytes,
                "reason": f"path={rel} size={actual} min={min_bytes}"}

    if ctype == "file_min_lines":
        try:
            min_lines = int(check.get("lines", 0))
        except (TypeError, ValueError):
            return {"type": ctype, "ok": False, "reason": "bad lines value"}
        if not abs_path.exists():
            return {"type": ctype, "ok": False, "reason": f"file not found: {rel}"}
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"type": ctype, "ok": False, "reason": f"read error: {e}"}
        actual = len([ln for ln in content.splitlines() if ln.strip()])
        return {"type": ctype, "ok": actual >= min_lines,
                "reason": f"path={rel} lines={actual} min={min_lines}"}

    return {"type": ctype, "ok": False, "reason": "unhandled check type"}


def _run_one_test(slug: str, t: dict) -> dict:
    cmd = (t.get("command") or "").strip()
    parts = cmd.split()
    if not parts:
        return {"name": t.get("name", "test"), "command": cmd, "ok": False,
                "rc": -1, "stdout": "", "stderr": "empty command", "checks": []}
    # Запускаем через venv-python если первая часть — python/python3 или pytest
    if parts[0].lower() in ("python", "python3"):
        res = run_with_project_python(slug, parts[1:], timeout=PHASE_TEST_TIMEOUT)
    elif parts[0].lower() == "pytest":
        res = run_with_project_python(slug, ["-m", "pytest", *parts[1:]], timeout=PHASE_TEST_TIMEOUT)
    else:
        res = run_in_project(slug, parts, timeout=PHASE_TEST_TIMEOUT)

    # Структурные checks из плана архитектора (P0): машинно-проверяемые условия.
    raw_checks = t.get("checks")
    check_results: list[dict] = []
    if isinstance(raw_checks, list) and raw_checks:
        for ch in raw_checks:
            check_results.append(_evaluate_check(slug, ch, res))
        checks_ok = all(c.get("ok") for c in check_results)
        # rc уже отдельная проверка только если её попросили; если её нет — 
        # требуем rc=0 неявно как минимальный sanity-check.
        has_rc_check = any(c.get("type") == "rc_zero" for c in check_results)
        rc_implicit_ok = True if has_rc_check else (res.get("returncode") == 0)
        overall_ok = checks_ok and rc_implicit_ok
        legacy_expects = ""
        legacy_expects_ok = True
    else:
        # Legacy fallback: свободнотекстовый expects ищется в stdout.
        # Сохранён ради обратной совместимости со старыми планами,
        # но архитектор больше не должен его генерировать.
        legacy_expects = t.get("expects") or ""
        legacy_expects_ok = (legacy_expects.lower() in (res.get("stdout", "")).lower()) if legacy_expects else True
        overall_ok = bool(res.get("ok")) and legacy_expects_ok

    return {
        "name":    t.get("name", "test"),
        "command": cmd,
        "ok":      overall_ok,
        "rc":      res.get("returncode"),
        "stdout":  (res.get("stdout") or "")[-400:],
        "stderr":  (res.get("stderr") or "")[-800:],
        "checks":  check_results,
        "expects": legacy_expects,
        "expects_ok": legacy_expects_ok,
    }


def _test(slug: str, plan: dict) -> list[dict]:
    out = []
    for t in (plan.get("tests") or []):
        rec = _run_one_test(slug, t)
        out.append(rec)
        add_phase(slug, f"test:{rec['name']}",
                  "ok" if rec["ok"] else "failed",
                  json.dumps({k: rec[k] for k in ("command","rc","expects_ok","stderr")}, ensure_ascii=False))
    return out


# ─── PHASE 6: heal ──────────────────────────────────────────────────────────
def _diagnose(spec: dict, file_paths: list[str], failed_test: dict, budget: Budget) -> dict:
    user = (
        f"Файлы проекта:\n" + "\n".join(f"  - {p}" for p in file_paths) + "\n\n"
        f"Тест упал:\n"
        f"  команда: {failed_test.get('command')}\n"
        f"  rc:      {failed_test.get('rc')}\n"
        f"  stderr:  {failed_test.get('stderr','')[:800]}\n"
        f"  stdout:  {failed_test.get('stdout','')[:400]}\n"
        f"  expects: {failed_test.get('expects','')}\n\n"
        f"Спецификация:\n{json.dumps(spec, ensure_ascii=False)[:1200]}\n"
    )
    raw = _llm(budget, MODEL_HEAVY, PROJECT_HEAL_SYSTEM, user,
               temperature=0.0, num_ctx=4096, where="heal.diagnose")
    diag = _safe_parse(raw)
    if not isinstance(diag, dict):
        diag = {}
    return diag


_MODNOTFOUND_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([A-Za-z0-9_.\-]+)['\"]")


# Сопоставление import-имени → PyPI-имя для очевидных расхождений
_IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
}


def _heal_missing_module(slug: str, failed: dict) -> dict | None:
    """Детерминистический healer для ModuleNotFoundError — без LLM.
    Выдергивает имя модуля из stderr и ставит его в venv.
    Возвращает dict с результатом или None если это не ModuleNotFoundError.
    """
    stderr = failed.get("stderr", "") or ""
    m = _MODNOTFOUND_RE.search(stderr)
    if not m:
        return None
    import_name = m.group(1).split(".")[0]  # берём корневой пакет
    pip_name = _IMPORT_TO_PIP.get(import_name, import_name)
    if not _PKG_PATTERN.match(pip_name):
        return {"ok": False, "missing": import_name, "reason": "unsafe pkg name"}
    res = pip_install(slug, [pip_name])
    return {
        "ok": bool(res.get("ok")),
        "missing": import_name,
        "installed_as": pip_name,
        "stderr": res.get("stderr", "")[-400:],
    }


def _heal_loop(slug: str, spec: dict, plan: dict, test_results: list[dict], budget: Budget) -> list[dict]:
    if all(r.get("ok") for r in test_results):
        return test_results
    file_paths = [f["path"] for f in plan["files"] if isinstance(f, dict) and "path" in f]

    for heal_iter in range(1, MAX_HEAL_ITERS + 1):
        failed = next((r for r in test_results if not r.get("ok")), None)
        if not failed:
            break

        # Быстрый путь: ModuleNotFoundError → детерминистический pip install (без LLM)
        miss = _heal_missing_module(slug, failed)
        if miss is not None:
            if miss.get("ok"):
                add_phase(slug, f"heal:iter{heal_iter}", "ok",
                          f"deterministic pip install {miss.get('installed_as')} (import {miss.get('missing')})")
                # Синхронизируем requirements.txt если он есть и пакета там нет
                try:
                    existing_files = get_project_files(slug)
                    if isinstance(existing_files, dict) and "requirements.txt" in existing_files:
                        cur = existing_files["requirements.txt"]
                        already = {ln.strip().lower() for ln in cur.splitlines() if ln.strip()}
                        if miss["installed_as"].lower() not in already:
                            new_req = (cur.rstrip() + "\n" + miss["installed_as"] + "\n").lstrip("\n")
                            write_project_file(slug, "requirements.txt", new_req)
                except Exception as e:
                    logger.debug(f"[heal] requirements.txt sync skipped: {e}")
                # Ретест без расхода LLM-бюджета
                test_results = _test(slug, plan)
                if all(r.get("ok") for r in test_results):
                    break
                continue
            else:
                add_phase(slug, f"heal:iter{heal_iter}", "failed",
                          f"deterministic pip install failed for {miss.get('missing')}: {miss.get('stderr','')[:200]}")
                # Не выходим — пусть LLM-ветка попробует другой фикс
        try:
            diag = _diagnose(spec, file_paths, failed, budget)
        except BudgetExceeded as e:
            add_phase(slug, f"heal:iter{heal_iter}", "failed", f"budget: {e}")
            break

        target_path = diag.get("target_file")
        if target_path not in file_paths:
            add_phase(slug, f"heal:iter{heal_iter}", "failed",
                      f"healer выбрал несуществующий файл: {target_path!r}")
            break
        fix_instr = diag.get("fix_instruction", "")
        if not fix_instr:
            add_phase(slug, f"heal:iter{heal_iter}", "failed", "пустая fix_instruction")
            break

        # Найти target dict в plan.files
        target_dict = next(f for f in plan["files"] if f.get("path") == target_path)
        existing = get_project_files(slug)
        try:
            current = read_project_file(slug, target_path)
        except Exception:
            current = ""
        feedback = (
            f"Тест '{failed.get('name')}' упал.\n"
            f"Диагноз: {diag.get('diagnosis','')}\n"
            f"Что нужно сделать: {fix_instr}\n"
            f"stderr: {failed.get('stderr','')[:600]}"
        )
        try:
            budget.check(f"heal:patch:iter{heal_iter}")
            new_code = coder_agent.patch_file(spec, plan, target_dict, current, feedback, existing=existing)
            budget.spend(1)
        except BudgetExceeded as e:
            add_phase(slug, f"heal:iter{heal_iter}", "failed", f"budget: {e}")
            break

        try:
            write_project_file(slug, target_path, new_code)
        except Exception as e:
            add_phase(slug, f"heal:iter{heal_iter}", "failed", f"write failed: {e}")
            break

        # перезапуск тестов
        new_results = _test(slug, plan)
        test_results = new_results
        add_phase(slug, f"heal:iter{heal_iter}", "ok",
                  f"target={target_path} all_ok={all(r['ok'] for r in new_results)}")
        if all(r.get("ok") for r in new_results):
            break
    return test_results


# ─── PHASE 7: README ────────────────────────────────────────────────────────
def _generate_readme(slug: str, spec: dict, plan: dict, test_results: list[dict], budget: Budget) -> str:
    summary_payload = {
        "title": spec.get("title"),
        "summary": spec.get("summary"),
        "requirements": spec.get("requirements", []),
        "acceptance_criteria": spec.get("acceptance_criteria", []),
        "files": [{"path": f.get("path"), "purpose": f.get("purpose","")} for f in plan["files"]],
        "tests": [
            {"name": r["name"], "command": r["command"], "ok": r["ok"], "expects": r.get("expects","")}
            for r in test_results
        ],
    }
    user = "Данные проекта в JSON:\n" + json.dumps(summary_payload, ensure_ascii=False, indent=2)
    try:
        raw = _llm(budget, MODEL_FAST, PROJECT_README_SYSTEM, user,
                   temperature=0.2, num_ctx=4096, where="readme")
    except BudgetExceeded:
        raw = ""
    md = _strip_json_fence(raw) if raw.strip().startswith("```") else (raw or "").strip()
    if not md:
        # Детерминированный fallback README — не падаем без LLM
        lines = [f"# {spec.get('title','Project')}", "", spec.get("summary",""), "", "## Структура"]
        for f in plan["files"]:
            lines.append(f"- `{f.get('path')}` — {f.get('purpose','')}")
        lines += ["", "## Запуск", "```"]
        for t in (plan.get("tests") or []):
            lines.append(t.get("command",""))
        lines += ["```", "", "## Проверки"]
        for crit in spec.get("acceptance_criteria", []):
            mark = "✅" if all(r["ok"] for r in test_results) else "⚠️"
            lines.append(f"- {crit} {mark}")
        md = "\n".join(lines) + "\n"
    try:
        write_project_file(slug, "README.md", md)
    except Exception as e:
        logger.warning(f"[project.readme] write failed: {e}")
    return md


# ─── PHASE 8: report ────────────────────────────────────────────────────────
def _report(slug: str, spec: dict, build_results: list[dict], test_results: list[dict],
            budget: Budget) -> str:
    summary = {
        "title":       spec.get("title"),
        "slug":        slug,
        "files":       [r["path"] for r in build_results if r.get("ok")],
        "build_ok":    sum(1 for r in build_results if r.get("ok")),
        "build_total": len(build_results),
        "tests_ok":    sum(1 for r in test_results if r.get("ok")),
        "tests_total": len(test_results),
        "first_test_error": next((r["stderr"] for r in test_results if not r["ok"] and r.get("stderr")), ""),
    }
    user = ("Итоги проекта в JSON:\n" + json.dumps(summary, ensure_ascii=False, indent=2)
            + f"\n\nПапка проекта: data/projects/{slug}/")
    try:
        return _llm(budget, MODEL_FAST, PROJECT_REPORT_SYSTEM, user,
                    temperature=0.3, num_ctx=2048, where="report").strip()
    except (BudgetExceeded, Exception) as e:
        logger.warning(f"[project.report] LLM unavailable: {e} — using deterministic")
        ok = summary["build_ok"]
        total = summary["build_total"]
        tn_ok = summary["tests_ok"]
        tn = summary["tests_total"]
        msg = (f"Проект {summary['title']} собран. Файлов: {ok} из {total}, "
               f"тестов прошло: {tn_ok} из {tn}. Лежит в data/projects/{slug}.")
        if summary["first_test_error"]:
            msg += " Есть ошибка в тестах — детали в manifest."
        return msg


# ─── PUBLIC: run() ──────────────────────────────────────────────────────────
def run(query: str, history: list[dict] | None = None,
        *, wall_budget_s: float = PROJECT_WALL_BUDGET_S,
        llm_budget: int = PROJECT_LLM_BUDGET) -> str:
    """Полный цикл: запрос → готовый проект."""
    if not isinstance(query, str) or not query.strip():
        return "Сэр, я не понял какой проект нужно сделать."

    budget = Budget(wall_s=wall_budget_s, llm=llm_budget)

    # PHASE 1
    try:
        spec = _intake(query, budget)
    except BudgetExceeded as e:
        return f"Бюджет исчерпан на этапе intake: {e}"
    except Exception as e:
        logger.error(f"[project.intake] {e}")
        return f"Не удалось разобрать задачу: {e}"

    try:
        manifest = create_project(spec)
    except Exception as e:
        logger.error(f"[project.create] {e}")
        return f"Не удалось создать проект: {e}"
    slug = manifest.slug
    # сохранить запрос
    try:
        m = load_manifest(slug); m.request = query[:500]; save_manifest(m)
    except Exception:
        pass
    add_phase(slug, "intake", "ok", spec.get("title", "")[:200])
    _set_last_phase(slug, "intake")

    return _continue(slug, budget, start_phase="architect")


def resume(slug: str, *, wall_budget_s: float = PROJECT_WALL_BUDGET_S,
           llm_budget: int = PROJECT_LLM_BUDGET) -> str:
    """Продолжить упавший проект с упавшей фазы."""
    try:
        m = load_manifest(slug)
    except Exception as e:
        return f"Не нашёл проект {slug}: {e}"
    last = m.last_phase or "intake"
    order = ["intake", "architect", "env", "build", "test", "heal", "readme", "finalize"]
    if last not in order:
        last = "intake"
    next_phase = order[order.index(last) + 1] if order.index(last) < len(order) - 1 else "finalize"
    budget = Budget(wall_s=wall_budget_s, llm=llm_budget)
    add_phase(slug, "resume", "ok", f"from={next_phase}")
    return _continue(slug, budget, start_phase=next_phase)


def _continue(slug: str, budget: Budget, *, start_phase: str) -> str:
    """Общая часть run() и resume(): фазы 2..8. При BudgetExceeded или фатальной
    ошибке ранний выход без финализации — чтобы last_phase остался на последней
    успешной, и resume(slug) мог продолжить."""
    m = load_manifest(slug)
    spec = m.spec
    plan = m.plan or {}
    build_results: list[dict] = []
    test_results:  list[dict] = []

    phases = ["architect", "env", "build", "test", "heal", "readme", "finalize"]
    if start_phase not in phases:
        start_phase = "architect"
    skip_until = phases.index(start_phase)

    def _abort_partial(reason: str) -> str:
        """Ранний выход: status=failed, без finalize-фазы, без _index, без README."""
        set_status(slug, "failed")
        _save_metrics(slug, partial=True, abort_reason=reason, **budget.summary())
        return f"Проект прерван: {reason}. Можно продолжить: resume({slug!r})."

    try:
        # PHASE 2: ARCHITECT
        if 0 >= skip_until:
            try:
                plan = _architect(spec, budget)
                m = load_manifest(slug); m.plan = plan; save_manifest(m)
                add_phase(slug, "architect", "ok",
                          f"files={len(plan['files'])} tests={len(plan.get('tests',[]))}")
                _set_last_phase(slug, "architect")
            except BudgetExceeded as e:
                add_phase(slug, "architect", "failed", f"budget: {e}")
                return _abort_partial(f"budget at architect: {e}")
            except Exception as e:
                add_phase(slug, "architect", "failed", str(e))
                _save_metrics(slug, **budget.summary())
                set_status(slug, "failed")
                return f"Не получилось спроектировать архитектуру: {e}"

        # PHASE 3: ENV
        if 1 >= skip_until:
            env_res = _phase_env(slug, plan)
            add_phase(slug, "env", "ok" if env_res["ok"] else "failed",
                      json.dumps(env_res, ensure_ascii=False)[:400])
            if env_res["ok"]:
                _set_last_phase(slug, "env")
            # env-failure не фатален: project может работать на stdlib

        # PHASE 4: BUILD
        if 2 >= skip_until:
            try:
                build_results = _build(slug, spec, plan, budget)
            except BudgetExceeded as e:
                add_phase(slug, "build", "failed", f"budget: {e}")
                return _abort_partial(f"budget at build: {e}")
            if build_results and any(r.get("ok") for r in build_results):
                _set_last_phase(slug, "build")
            elif build_results and not any(r.get("ok") for r in build_results):
                # Все файлы упали (обычно при исчерпании бюджета внутри _build) —
                # выходим без finalize, чтобы last_phase остался на architect/env
                # и resume(slug) мог корректно продолжить с build.
                add_phase(slug, "build", "failed", "no successful files")
                return _abort_partial("build had no successful files")

        # PHASE 5: TEST
        if 3 >= skip_until:
            try:
                test_results = _test(slug, plan)
                _set_last_phase(slug, "test")
            except Exception as e:
                add_phase(slug, "test", "failed", str(e))
                test_results = []

        # PHASE 6: HEAL
        if 4 >= skip_until and test_results and not all(r.get("ok") for r in test_results):
            try:
                test_results = _heal_loop(slug, spec, plan, test_results, budget)
                _set_last_phase(slug, "heal")
            except BudgetExceeded as e:
                add_phase(slug, "heal", "failed", f"budget: {e}")
                # heal-budget — не фатально, идём в README/finalize с тем что есть

        # PHASE 7: README
        if 5 >= skip_until:
            try:
                _generate_readme(slug, spec, plan, test_results, budget)
                add_phase(slug, "readme", "ok", "README.md generated")
                _set_last_phase(slug, "readme")
            except Exception as e:
                add_phase(slug, "readme", "failed", str(e))

        # PHASE 8: FINALIZE
        all_files_ok = all(r.get("ok") for r in build_results) if build_results else False
        all_tests_ok = all(r.get("ok") for r in test_results)  if test_results  else True
        final = "done" if (all_files_ok and all_tests_ok) else "failed"
        set_status(slug, final)
        _save_metrics(slug, **budget.summary())
        add_phase(slug, "finalize", "ok", final)
        _set_last_phase(slug, "finalize")

        append_index_record({
            "slug":     slug,
            "title":    spec.get("title"),
            "status":   final,
            "files":    len([r for r in build_results if r.get("ok")]),
            "tests_ok": sum(1 for r in test_results if r.get("ok")),
            "tests_total": len(test_results),
            "metrics":  budget.summary(),
        })

        return _final_report(slug, spec, build_results, test_results, budget)

    except Exception as e:
        logger.exception(f"[project] unexpected: {e}")
        set_status(slug, "failed")
        _save_metrics(slug, **budget.summary())
        return f"Ошибка проекта: {e}. Можно продолжить через resume({slug})."


def _final_report(slug: str, spec: dict, build_results: list[dict],
                  test_results: list[dict], budget: Budget) -> str:
    return _report(slug, spec, build_results, test_results, budget)


# ─── CLI ────────────────────────────────────────────────────────────────────
def _main() -> int:
    p = argparse.ArgumentParser(description="Jarvis ProjectAgent — Level 4")
    p.add_argument("query", nargs="?", help="запрос на проект")
    p.add_argument("--resume", metavar="SLUG", help="продолжить упавший проект")
    p.add_argument("--list",   action="store_true", help="показать все проекты")
    p.add_argument("--wall",   type=int, default=PROJECT_WALL_BUDGET_S, help="wall-clock budget (s)")
    p.add_argument("--llm",    type=int, default=PROJECT_LLM_BUDGET, help="LLM-call budget")
    args = p.parse_args()

    if args.list:
        for r in list_projects():
            print(f"{r['slug']:40s} {r['status']:12s} {r.get('title','')}")
        return 0
    if args.resume:
        out = resume(args.resume, wall_budget_s=args.wall, llm_budget=args.llm)
        print(out)
        return 0
    if not args.query:
        p.print_help(sys.stderr)
        return 2
    out = run(args.query, [], wall_budget_s=args.wall, llm_budget=args.llm)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
