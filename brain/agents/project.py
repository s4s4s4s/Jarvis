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
import hashlib
import re
import sys
import time
from dataclasses import asdict
from typing import Any

from brain.client import (
    chat,
    MODEL_FAST,
    MODEL_HEAVY,
    MODEL_INTAKE,
    MODEL_ARCHITECT,
    MODEL_HEALER,
    MODEL_README,
    MODEL_REPORT,
)
from brain.prompts import (
    PROJECT_INTAKE_SYSTEM,
    PROJECT_ARCHITECT_SYSTEM,
    PROJECT_REPORT_SYSTEM,
    PROJECT_HEAL_SYSTEM,
    PROJECT_README_SYSTEM,
)
from brain.agents import coder as coder_agent
from brain.agents import reviewer as reviewer_agent
from brain.agents import aider_runner
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
    run_shell_in_project,
    _has_shell_metachars,
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
MAX_HEAL_ITERS   = 4  # P9.6: было 2 — расширили для detmin+aider цепочки (rss_parser требовал 3+)
MAX_FILES        = 10
PHASE_TEST_TIMEOUT = 30
PROJECT_WALL_BUDGET_S = 600       # 10 минут на проект целиком
PROJECT_LLM_BUDGET    = 40        # суммарно на все фазы

# P3: адаптивный бюджет по размеру проекта. Не выбирает решения, только сколько раз попробовать.
BUDGET_TIERS = {
    "XS": {"wall_s": 180,  "llm": 15},   # 1 файл, однострочный запрос
    "S":  {"wall_s": 360,  "llm": 30},   # 1-2 файла, парсер/скрипт
    "M":  {"wall_s": 600,  "llm": 60},   # 3-4 файла, интеграция
    "L":  {"wall_s": 1200, "llm": 120},  # 5+ файлов, сложный проект
}


def estimate_complexity(query: str, spec: dict | None = None, plan: dict | None = None) -> str:
    """P3: возвращает тир бюджета (XS/S/M/L) по размерным метрикам.

    Правила:
      - Число файлов в плане (если есть) — главный сигнал: 1→XS, 2→S, 3-4→M, 5+→L.
      - Без плана: по длине запроса и числу requirements.
      - Никакого выбора «по ключевым словам» — только размерные метрики.
    """
    files_n = 0
    if isinstance(plan, dict):
        files = plan.get("files") or []
        if isinstance(files, list):
            files_n = sum(1 for f in files if isinstance(f, dict) and f.get("path"))
    if files_n >= 5:
        return "L"
    if files_n in (3, 4):
        return "M"
    if files_n == 2:
        return "S"
    if files_n == 1:
        return "XS"

    text = (query or "").strip()
    word_count = len(text.split())
    req_count = 0
    if isinstance(spec, dict):
        reqs = spec.get("requirements") or []
        if isinstance(reqs, list):
            req_count = len(reqs)
    if word_count <= 7 and req_count <= 1:
        return "XS"
    if word_count >= 80 or req_count >= 5:
        return "L"
    if word_count >= 40 or req_count >= 3:
        return "M"
    return "S"


def budget_for_tier(tier: str) -> dict:
    """Параметры бюджета для тира. Дефолт — M."""
    return BUDGET_TIERS.get(tier, BUDGET_TIERS["M"])


# ─── budget tracking ──────────────────────────────────────────────────────────────────────────────
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


# ─── helpers ───────────────────────────────────────────────────────────────────────────────────
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


# ─── PHASE 1: intake ──────────────────────────────────────────────────────────────────────────────
def _normalize_intake_spec(spec: dict, query: str) -> dict:
    """P6: пост-обработка spec от LLM — структурные гарантии, без ключевых слов."""
    from tools.projects import slugify

    if not isinstance(spec, dict):
        spec = {}

    # title — ПЛОХИЕ значения (плейсхолдеры из промпта) отбрасываем
    title = str(spec.get("title") or "").strip()
    bad_titles = {"untitled-project", "untitled", "project", "unnamed", "name", "название", "проект"}
    if not title or len(title) > 100 or title.lower() in bad_titles:
        words = (query or "").strip().split()[:5]
        title = " ".join(words) if words else "untitled-project"
    spec["title"] = title

    # slug — обязательно непустой и валидный
    slug = str(spec.get("slug") or "").strip().lower()
    if not slug or not re.match(r"^[a-z0-9\-]{1,40}$", slug):
        slug = slugify(title)
    if not slug:
        slug = "untitled-project"
    spec["slug"] = slug[:40]

    # kind / language — разрешённые множества
    allowed_kind = {"script", "cli", "web", "bot", "library", "data"}
    kind = str(spec.get("kind") or "").strip().lower()
    spec["kind"] = kind if kind in allowed_kind else "script"

    allowed_lang = {"python", "javascript", "typescript", "html", "other"}
    lang = str(spec.get("language") or "").strip().lower()
    spec["language"] = lang if lang in allowed_lang else "python"

    # summary — если ровно равен запросу (эхо), берём первые ~15 слов
    summary = str(spec.get("summary") or "").strip()
    q_norm_full = (query or "").strip()
    if not summary:
        summary = q_norm_full[:200]
    elif summary.lower() == q_norm_full.lower():
        words = q_norm_full.split()[:15]
        summary = " ".join(words) + ("…" if len(q_norm_full.split()) > 15 else "")
    spec["summary"] = summary[:300]

    # requirements: список строк; флаг эхо если LLM выдала весь query одним пунктом
    reqs_raw = spec.get("requirements") or []
    if not isinstance(reqs_raw, list):
        reqs_raw = [str(reqs_raw)]
    reqs = [str(r).strip() for r in reqs_raw if str(r).strip()]
    q_norm = (query or "").strip().lower()
    is_echo = (
        len(reqs) == 1
        and q_norm
        and reqs[0].lower().startswith(q_norm[: min(40, len(q_norm))])
    )
    # P6.1: если эхо — разбиваем примитивно по знакам препинания. Не заменяет LLM,
    # но хотя бы даёт архитектору отдельные куски вместо одной слипшейся строки.
    if is_echo:
        spec["_intake_warning"] = "requirements is echo of query"
        # Разбиваем по , ; и « и »
        chunks = re.split(r"[,;]|\s+и\s+|\s+и\s+", reqs[0])
        chunks = [c.strip(" .!?—-") for c in chunks if c.strip(" .!?—-")]
        if len(chunks) >= 2:
            reqs = chunks[:6]
    spec["requirements"] = reqs

    # deliverables
    dlv_raw = spec.get("deliverables") or []
    if not isinstance(dlv_raw, list):
        dlv_raw = [str(dlv_raw)]
    spec["deliverables"] = [str(d).strip() for d in dlv_raw if str(d).strip()] or ["main.py"]

    # acceptance_criteria
    ac_raw = spec.get("acceptance_criteria") or []
    if not isinstance(ac_raw, list):
        ac_raw = [str(ac_raw)]
    spec["acceptance_criteria"] = (
        [str(a).strip() for a in ac_raw if str(a).strip()]
        or ["скрипт запускается без ошибок"]
    )

    return spec


def _intake(query: str, budget: Budget) -> dict:
    raw = _llm(budget, MODEL_INTAKE, PROJECT_INTAKE_SYSTEM, query,
               temperature=0.1, num_ctx=4096, where="intake")
    spec = _safe_parse(raw)
    if not isinstance(spec, dict) or not spec.get("title"):
        # P6: пустые requirements (не эхо запроса)
        spec = {
            "title": " ".join((query or "").strip().split()[:5]) or "untitled-project",
            "slug": "",
            "kind": "script",
            "language": "python",
            "summary": (query or "")[:200],
            "requirements": [],
            "deliverables": ["main.py"],
            "acceptance_criteria": ["скрипт запускается без ошибок"],
        }
    return _normalize_intake_spec(spec, query)


# ─── PHASE 2: architect ──────────────────────────────────────────────────────────────────────────────
def _architect(spec: dict, budget: Budget) -> dict:
    user = "Спецификация проекта:\n" + json.dumps(spec, ensure_ascii=False, indent=2)
    raw = _llm(budget, MODEL_ARCHITECT, PROJECT_ARCHITECT_SYSTEM, user,
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


# ─── PHASE 3: env (venv + pip) ──────────────────────────────────────────────────────────────────────────────────────
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


# ─── PHASE 4: build (coder ↔ reviewer loop) ──────────────────────────────────────────────────────────────────────────
I need to send the full file content but it's 145KB which is too large for a single tool call parameter. Let me use push_files with the actual file read from disk.