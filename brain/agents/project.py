"""
brain/agents/project.py — ProjectAgent (Level 4: Создатель).

Принимает запрос пользователя естественным языком и ведёт проект от спеки до
работающих файлов с тестами:

  1. INTAKE     — извлекает структурированную спеку (LLM)
  2. ARCHITECT  — проектирует файлы и тесты (LLM)
  3. BUILD      — для каждого файла: Coder пишет → Reviewer критикует →
                  Coder правит. До MAX_REVIEW_ITERS итераций.
  4. TEST       — выполняет команды тестов из плана внутри песочницы проекта
  5. REPORT     — короткий устный итог пользователю

Никогда не падает молча. На каждом шаге пишет фазу в manifest.json и
journal-строку в logs/projects.jsonl.

Выход: строка-ответ для озвучки. Подробности — в data/projects/<slug>/manifest.json.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from brain.client import chat, MODEL_FAST, MODEL_HEAVY
from brain.prompts import (
    PROJECT_INTAKE_SYSTEM,
    PROJECT_ARCHITECT_SYSTEM,
    PROJECT_REPORT_SYSTEM,
)
from brain.agents import coder as coder_agent
from brain.agents import reviewer as reviewer_agent
from tools.projects import (
    create_project,
    write_project_file,
    read_project_file,
    add_phase,
    set_status,
    save_manifest,
    load_manifest,
    run_in_project,
    python_smoke,
)

logger = logging.getLogger(__name__)

MAX_REVIEW_ITERS = 2          # цикл write↔review на файл
MAX_FILES        = 10
PHASE_TEST_TIMEOUT = 30


# ─── helpers ─────────────────────────────────────────────────────────────────
def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
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


def _llm(model: str, system: str, user: str, *, temperature: float = 0.1, num_ctx: int = 4096) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return chat(model, msgs, options={"temperature": temperature, "num_ctx": num_ctx})


# ─── PHASE 1: intake ─────────────────────────────────────────────────────────
def _intake(query: str) -> dict:
    raw = _llm(MODEL_FAST, PROJECT_INTAKE_SYSTEM, query, temperature=0.1, num_ctx=4096)
    spec = _safe_parse(raw)
    if not isinstance(spec, dict) or not spec.get("title"):
        # Минимальный fallback: лучше двинуться дальше с черновой спекой,
        # чем уронить пайплайн. ReviewerAgent скорее всего попросит revise.
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


# ─── PHASE 2: architect ──────────────────────────────────────────────────────
def _architect(spec: dict) -> dict:
    user = (
        "Спецификация проекта:\n"
        + json.dumps(spec, ensure_ascii=False, indent=2)
    )
    raw = _llm(MODEL_HEAVY, PROJECT_ARCHITECT_SYSTEM, user, temperature=0.1, num_ctx=8192)
    plan = _safe_parse(raw)
    files = plan.get("files") or []
    if not isinstance(files, list) or not files:
        plan = {
            "files": [{"path": "main.py", "purpose": "точка входа", "depends_on": ["stdlib"]}],
            "build_steps": [
                {"step": 1, "kind": "create_file", "target": "main.py", "description": "создаём entry"},
                {"step": 2, "kind": "smoke_run",   "target": "python main.py", "description": "smoke"},
            ],
            "tests": [{"name": "smoke", "command": "python main.py", "expects": "вывод без исключения"}],
        }
    # safety: ограничим число файлов
    plan["files"] = (plan.get("files") or [])[:MAX_FILES]
    plan.setdefault("build_steps", [])
    plan.setdefault("tests", [])
    return plan


# ─── PHASE 3: build (coder ↔ reviewer loop) ─────────────────────────────────
def _build_one_file(slug: str, spec: dict, plan: dict, target: dict) -> dict:
    """
    Generate one file with up to MAX_REVIEW_ITERS write↔review cycles.
    Returns summary dict for phase log.
    """
    feedback = ""
    code = ""
    final_review: dict[str, Any] = {}
    for it in range(MAX_REVIEW_ITERS + 1):
        if it == 0:
            code = coder_agent.write_file(spec, plan, target)
        else:
            code = coder_agent.patch_file(spec, plan, target, code, feedback)

        rv = reviewer_agent.review(spec, target, code)
        final_review = rv
        if rv["verdict"] == "approve":
            break
        feedback = reviewer_agent.issues_as_feedback(rv["issues"])
        if not feedback:
            # reviewer сказал revise но без issues — выходим
            break

    # запишем итоговый файл
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
    }


def _build(slug: str, spec: dict, plan: dict) -> list[dict]:
    results = []
    for target in plan["files"]:
        if not isinstance(target, dict) or "path" not in target:
            continue
        res = _build_one_file(slug, spec, plan, target)
        results.append(res)
        status = "ok" if res.get("ok") else "failed"
        add_phase(slug, f"build:{target.get('path')}", status, json.dumps(res, ensure_ascii=False))
    return results


# ─── PHASE 4: test ───────────────────────────────────────────────────────────
def _test(slug: str, plan: dict) -> list[dict]:
    out = []
    tests = plan.get("tests") or []
    for t in tests:
        cmd = (t.get("command") or "").strip()
        if not cmd:
            continue
        # принимаем только простые команды через split. shell=False гарантирован в run_in_project.
        # для python отдадим через python_smoke (он использует sys.executable — корректно вне зависимости от ОС).
        parts = cmd.split()
        if parts and parts[0].lower() in ("python", "python3"):
            entry = parts[1] if len(parts) > 1 else "main.py"
            res = python_smoke(slug, entry, timeout=PHASE_TEST_TIMEOUT)
        else:
            res = run_in_project(slug, parts, timeout=PHASE_TEST_TIMEOUT)
        rec = {
            "name":    t.get("name", "test"),
            "command": cmd,
            "ok":      res.get("ok", False),
            "rc":      res.get("returncode"),
            "stderr":  (res.get("stderr") or "")[-400:],
        }
        out.append(rec)
        add_phase(
            slug,
            f"test:{rec['name']}",
            "ok" if rec["ok"] else "failed",
            json.dumps(rec, ensure_ascii=False),
        )
    return out


# ─── PHASE 5: report ─────────────────────────────────────────────────────────
def _report(slug: str, spec: dict, build_results: list[dict], test_results: list[dict]) -> str:
    summary = {
        "title":  spec.get("title"),
        "slug":   slug,
        "files":  [r["path"] for r in build_results if r.get("ok")],
        "build_ok":  sum(1 for r in build_results if r.get("ok")),
        "build_total": len(build_results),
        "tests_ok":  sum(1 for r in test_results if r.get("ok")),
        "tests_total": len(test_results),
        "first_test_error": next((r["stderr"] for r in test_results if not r["ok"] and r.get("stderr")), ""),
    }
    user = (
        "Итоги проекта в JSON:\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + f"\n\nПапка проекта: data/projects/{slug}/"
    )
    try:
        return _llm(MODEL_FAST, PROJECT_REPORT_SYSTEM, user, temperature=0.3, num_ctx=2048).strip()
    except Exception as e:
        logger.error(f"[project.report] LLM error: {e}")
        # дет fallback: соберём короткий русский текст руками
        ok = summary["build_ok"]
        total = summary["build_total"]
        tn_ok = summary["tests_ok"]
        tn = summary["tests_total"]
        msg = f"Проект {summary['title']} собран. Файлов: {ok} из {total}, тестов прошло: {tn_ok} из {tn}. Лежит в data/projects/{slug}."
        if summary["first_test_error"]:
            msg += " Есть ошибка в тестах — детали в manifest."
        return msg


# ─── PUBLIC: run() ───────────────────────────────────────────────────────────
def run(query: str, history: list[dict] | None = None) -> str:
    """
    Entry-point вызывается из brain/ask.py при route=project.
    Возвращает строку для озвучки.
    """
    if not isinstance(query, str) or not query.strip():
        return "Сэр, я не понял какой проект нужно сделать."

    # 1. INTAKE
    try:
        spec = _intake(query)
    except Exception as e:
        logger.error(f"[project.intake] {e}")
        return f"Не удалось разобрать задачу проекта: {e}"

    # создаём папку и манифест
    try:
        manifest = create_project(spec)
    except Exception as e:
        logger.error(f"[project.create] {e}")
        return f"Не удалось создать проект: {e}"

    slug = manifest.slug
    add_phase(slug, "intake", "ok", spec.get("title", "")[:200])

    # 2. ARCHITECT
    try:
        plan = _architect(spec)
        m = load_manifest(slug)
        m.plan = plan
        save_manifest(m)
        add_phase(slug, "architect", "ok", f"files={len(plan['files'])} tests={len(plan.get('tests',[]))}")
    except Exception as e:
        logger.error(f"[project.architect] {e}")
        add_phase(slug, "architect", "failed", str(e))
        set_status(slug, "failed")
        return f"Сэр, не получилось спроектировать архитектуру: {e}"

    # 3. BUILD
    try:
        build_results = _build(slug, spec, plan)
    except Exception as e:
        logger.error(f"[project.build] {e}")
        add_phase(slug, "build", "failed", str(e))
        set_status(slug, "failed")
        return f"Сборка проекта сорвалась: {e}"

    # 4. TEST
    try:
        test_results = _test(slug, plan)
    except Exception as e:
        logger.error(f"[project.test] {e}")
        add_phase(slug, "test", "failed", str(e))
        test_results = []

    # 5. финальный статус и отчёт
    all_files_ok = all(r.get("ok") for r in build_results) if build_results else False
    all_tests_ok = all(r.get("ok") for r in test_results) if test_results else True
    final = "done" if (all_files_ok and all_tests_ok) else "failed"
    set_status(slug, final)
    add_phase(slug, "finalize", "ok", final)

    return _report(slug, spec, build_results, test_results)
