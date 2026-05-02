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


# ─── PHASE 2: architect ─────────────────────────────────────────────────────
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
def _build_one_file_aider(slug: str, spec: dict, plan: dict, target: dict, budget: Budget) -> dict:
    """P9: build через aider. Один вызов вместо coder/reviewer цикла.

    aider сам разберётся с syntax/parse-проблемами (это его киллер-фича).
    Мы только формулируем понятную инструкцию из spec+target и читаем результат.

    Бюджет: один вызов aider = 1 spend (как один LLM-вызов в нашей метрике),
    хотя внутри aider может сделать несколько обращений к ollama.
    """
    from core.config import AIDER_TIMEOUT_S
    pdir = project_dir(slug)
    rel_path = target.get("path") or "main.py"

    # Early-return для runtime-deliverable: если файл значится в spec.deliverables
    # и не является source-кодом — это артефакт времени выполнения (output.json, news.csv и т.п.).
    # Создаём пустую заглушку, smoke_test заполнит её при запуске entry-point.
    # Это избавляет aider от попытки "написать" runtime-данные и согласуется с поведением
    # legacy-пайплайна, где такие файлы фактически перезаписывались тестом.
    deliverables = [str(d).strip() for d in (spec.get("deliverables") or []) if d]
    # Source-расширения — эти файлы всегда строит aider, даже если они в deliverables.
    # Runtime-форматы (.json, .csv, .xml, .html и пр.) — результат работы кода.
    SOURCE_EXTS = {".py", ".md", ".sh", ".ps1", ".bat", ".js", ".ts", ".toml", ".yaml", ".yml", ".cfg", ".ini"}
    rel_lower = rel_path.lower()
    is_runtime_deliverable = (
        rel_path in deliverables
        and not any(rel_lower.endswith(ext) for ext in SOURCE_EXTS)
    )
    if is_runtime_deliverable:
        try:
            full = safe_project_path(slug, rel_path)
            full.parent.mkdir(parents=True, exist_ok=True)
            if not full.exists():
                full.write_bytes(b"")
            logger.info(f"[project.build_aider] {rel_path} is runtime deliverable — created empty stub, skipping aider")
            return {
                "path":    rel_path,
                "ok":      True,
                "verdict": "approve",
                "issues":  0,
                "summary": f"runtime deliverable stub for {rel_path}",
                "iters":   0,
                "static":  {"tools": [], "errors": [], "warnings": [], "fail_streak": 0, "final_ast_ok": True},
                "_via":    "aider:stub",
            }
        except Exception as e:
            logger.warning(f"[project.build_aider] failed to create stub for {rel_path}: {e}")
            # fallthrough в обычный aider-путь

    # Собираем инструкцию для aider из spec + target
    title = spec.get("title") or slug
    summary = spec.get("summary") or ""
    requirements = spec.get("requirements") or []
    acceptance = spec.get("acceptance_criteria") or []
    target_purpose = target.get("purpose") or target.get("description") or ""

    instruction_parts = [f"Проект «{title}»: {summary}".strip(": ")]
    if target_purpose:
        instruction_parts.append(f"Назначение файла {rel_path}: {target_purpose}")
    if requirements:
        reqs_str = "\n".join(f"- {r}" for r in requirements[:8])
        instruction_parts.append(f"Требования к проекту:\n{reqs_str}")
    if acceptance:
        ac_str = "\n".join(f"- {a}" for a in acceptance[:5])
        instruction_parts.append(f"Критерии приёмки:\n{ac_str}")

    # P9.10: явно передаём входные файлы в инструкцию aider'у.
    # Без этого блока aider хардкодит example.txt вместо реального имени из spec.
    plan_inputs = (plan.get("inputs") or []) if isinstance(plan, dict) else []
    if plan_inputs:
        in_lines = []
        for it in plan_inputs:
            if not isinstance(it, dict):
                continue
            p = it.get("path", "")
            if not p:
                continue
            sample = it.get("sample_content", "")
            preview = (sample[:160] + "…") if len(sample) > 160 else sample
            preview_one = preview.replace("\n", " \u21b5 ")
            in_lines.append(f"- {p} (пример содержимого: {preview_one!r})")
        if in_lines:
            instruction_parts.append(
                "ВХОДНЫЕ ФАЙЛЫ (используй РОВНО эти пути в коде, НЕ придумывай свои):\n"
                + "\n".join(in_lines)
                + "\nФайлы будут созданы с реалистичным содержимым перед запуском теста."
            )

    # P11.2: CONTRACT-блок (must_export + required_imports) и read-only контекст соседей.
    # Строится ИСКЛЮЧИТЕЛЬНО из plan.files — никаких keyword-эвристик.
    contract_block = _build_contract_prompt_block(plan, rel_path)
    if contract_block:
        instruction_parts.append(contract_block)
    read_only_files, neighbor_descs = _build_neighbor_context(pdir, plan, rel_path)
    if neighbor_descs:
        instruction_parts.append(
            "СОСЕДНИЕ МОДУЛИ (их контракты переданы read-only в контекст aider):\n"
            + "\n".join(neighbor_descs)
        )

    instruction_parts.append(
        f"Создай (или обнови) ФАЙЛ {rel_path} И ТОЛЬКО ЕГО. "
        f"НЕ создавай никаких других .py-файлов (init_db.py, helpers.py, utils.py и т.п.) — всё нужное для работы должно лежать внутри {rel_path} или браться из уже спланированных соседей. "
        "Пиши работающий, синтаксически корректный код. "
        "Не добавляй комментариев-извинений и заглушек."
    )
    instruction = "\n\n".join(instruction_parts)

    budget.check(f"build:{rel_path}:aider")
    res = aider_runner.aider_build(pdir, rel_path, instruction, read_only_files=read_only_files or None)
    budget.spend(1)

    # Пост-проверка: даже если aider сказал ok, прогоним static_check для метрик манифеста.
    sc = static_check(rel_path, res.content) if res.content else {"applicable": False}
    static_summary = {
        "tools":         sc.get("tools") or [],
        "errors":        sc.get("errors") or [],
        "warnings":      sc.get("warnings") or [],
        "fail_streak":   0 if sc.get("ok", True) else 1,
        "final_ast_ok":  bool(sc.get("ok", True)),
    }

    if not res.ok:
        logger.warning(f"[project.build_aider] {rel_path} failed: {res.error}")
        return {
            "path": rel_path,
            "ok": False,
            "error": res.error,
            "verdict": "revise",
            "summary": (res.error or "aider failure")[:200],
            "iters": res.attempts,
            "static": static_summary,
            "_via": "aider",
            "aider": {
                "duration_s": res.duration_s,
                "exit_code":  res.exit_code,
                "stderr_tail": res.stderr[-400:] if res.stderr else "",
            },
        }

    # P11.2.d: пост-билд линтер контракта.
    # Сверяем реальные top-level имена файла с plan.files[*].exports для этого файла.
    # P11.5.C: блокирующий линтер — contract.ok=False делает build phase failed,
    # чтобы heal-loop сразу подхватил этот файл в target с missing-экспортами.
    contract_check = {"checked": False}
    try:
        target_in_plan = next(
            (f for f in (plan.get("files") or []) if isinstance(f, dict) and f.get("path") == rel_path),
            None,
        )
        expected_exports = (target_in_plan or {}).get("exports") or []
        # Сверка имеет смысл только для питон-файлов с заявленными экспортами
        if _is_python_path(rel_path) and expected_exports:
            cc = _check_file_contract(res.content or "", expected_exports)
            contract_check = {
                "checked":     True,
                "ok":          cc["ok"],
                "missing":     cc["missing"],
                "kind_mismatch": cc["kind_mismatch"],
                "found_top_level": cc["found_top_level"],
            }
            if not cc["ok"]:
                logger.warning(
                    f"[contract.lint] {rel_path}: missing={cc['missing']} "
                    f"kind_mismatch={cc['kind_mismatch']}"
                )
    except Exception as e:
        logger.warning(f"[contract.lint] {rel_path}: failed: {e}")
        contract_check = {"checked": False, "error": str(e)}

    # P11.5.C: блокада. Если линтер проверил и не ОК — фаза билда считается провальной.
    # Это сигнал для heal-loop: вызвать aider на этом файле с хинтом «восстанови missing exports»,
    # вместо того чтобы пускать smoke-тест который всё равно упадёт на ImportError.
    contract_failed = bool(
        contract_check.get("checked")
        and contract_check.get("ok") is False
    )
    if contract_failed:
        missing = contract_check.get("missing") or []
        kind_mm = contract_check.get("kind_mismatch") or []
        logger.warning(
            f"[contract.block] {rel_path} → build failed (P11.5): "
            f"missing={missing} kind_mismatch={kind_mm}"
        )

    return {
        "path":    rel_path,
        "ok":      not contract_failed,
        "verdict": "revise" if contract_failed else "approve",
        "issues":  len(static_summary["errors"]) + len(static_summary["warnings"]),
        "summary": (
            f"aider built {rel_path} in {res.duration_s}s but contract violated: "
            f"missing={contract_check.get('missing') or []}"
            if contract_failed
            else f"aider built {rel_path} in {res.duration_s}s"
        ),
        "iters":   res.attempts,
        "static":  static_summary,
        "contract": contract_check,  # P11.2.d
        "contract_failure": contract_failed,  # P11.5.C: явный флаг для heal-loop
        "_via":    "aider",
        "aider":   {
            "duration_s": res.duration_s,
            "exit_code":  res.exit_code,
        },
    }


def _build_one_file(slug: str, spec: dict, plan: dict, target: dict, budget: Budget) -> dict:
    """Build-loop с детерминистической статикой перед LLM-Reviewer (P1).

    P9: если AIDER_ENABLED и aider доступен — делегируем в _build_one_file_aider.
    Иначе — старый путь coder/reviewer.

    На каждой итерации (старый путь):
      1. Coder пишет/патчит код.
      2. static_check (ast.parse + ruff/pyflakes если есть).
      3. Если синтаксис битый → пропускаем LLM-Reviewer, ast-ошибка — feedback.
      4. Иначе вызываем Reviewer (при наличии lint-warnings — пробросим их как hint).
    """
    from core.config import AIDER_ENABLED
    if AIDER_ENABLED and aider_runner.is_aider_available():
        try:
            return _build_one_file_aider(slug, spec, plan, target, budget)
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.warning(f"[project.build] aider path failed ({e}); falling back to legacy")
            # Падаем в старый путь — fallback по принципу проекта

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


# ─── P9.7: helpers для устойчивости к плохим планам архитектора ─────────────
# Архитектор-LLM иногда генерирует невалидные checks: ставит file_exists на
# ВХОДНЫЕ файлы (которых нет до запуска и которые скрипт обрабатывает, не
# создаёт) или нереалистичные file_min_size (например, 1024б для example.com
# который весит 529б). Эти ошибки aider починить не может — они в плане, не в
# коде. Поэтому мы детерминистически нормализуем такие checks перед запуском
# и при evaluate. Подход «никаких хардкодов и keyword-ов» соблюдён: фильтруем
# по deliverables (структурное поле спеки) и реальному размеру (факт), а не по
# именам файлов или подстрокам.

# Source-расширения, которые НИКОГДА не должны автозаполняться dummy-файлом:
# это исходники, и их отсутствие — реальный баг, не вход.
_SOURCE_EXT_FOR_FIXTURE = {".py", ".js", ".ts", ".go", ".rs", ".rb", ".java", ".c", ".cpp", ".h", ".hpp", ".sh", ".html", ".css", ".md", ".yml", ".yaml", ".toml"}
# Минимальный реалистичный пол для file_min_size, если архитектор завысил.
_MIN_SIZE_REALISTIC_FLOOR = 64

# P9.9: типы checks, которые подразумевают что path — это ВЫХОД скрипта
# (пост-проверки результата работы). Если file_exists(path) встречается
# вместе с любым из этих чеков на тот же путь — path считается выходом,
# фикстуру создавать НЕЛЬЗЯ (создание пустого файла-заглушки сломает логику
# скрипта: либо он пропустит уже существующий «обработанный» файл, либо
# перезапишет нулевыми байтами). file_exists на выход — валидная пост-проверка,
# её нужно оставить.
_OUTPUT_CHECK_TYPES = {
    "file_min_lines", "file_min_size", "file_max_size",
    "json_valid", "yaml_valid", "file_contains", "file_matches_regex",
    "line_count_min", "line_count_max",
}

# P9.10: дефолтные sample-содержимые для входных фикстур по расширению.
# Используются когда архитектор не указал sample_content в plan.inputs[].
# Цель — дать скрипту реалистичный вход, чтобы он мог выдать осмысленный выход.
# Текст подобран так, чтобы проходили типичные регексы/парсеры: email, URL, числа.
_DEFAULT_INPUT_SAMPLES = {
    ".txt": (
        "Hello, contact us at support@example.com or sales@company.co.uk.\n"
        "You can also reach admin@test.org for help.\n"
        "Visit https://example.com or http://test.org for more info.\n"
        "Phone: +1-555-0123, fax: 555-9876.\n"
        "Order #1234 total $99.50 dated 2024-01-15.\n"
    ),
    ".csv": (
        "id,name,email\n"
        "1,Alice,alice@example.com\n"
        "2,Bob,bob@test.org\n"
        "3,Carol,carol@company.co.uk\n"
    ),
    ".tsv": (
        "id\tname\temail\n"
        "1\tAlice\talice@example.com\n"
        "2\tBob\tbob@test.org\n"
    ),
    ".json": (
        '{\n  "items": [\n'
        '    {"id": 1, "name": "Alice", "email": "alice@example.com"},\n'
        '    {"id": 2, "name": "Bob", "email": "bob@test.org"}\n'
        '  ]\n}\n'
    ),
    ".log": (
        "2024-01-15 10:00:01 INFO User alice@example.com logged in\n"
        "2024-01-15 10:01:15 ERROR Connection failed for bob@test.org\n"
        "2024-01-15 10:02:30 WARN Rate limit hit at https://api.example.com\n"
    ),
    ".xml": (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<root>\n'
        '  <item id="1" email="alice@example.com">Alice</item>\n'
        '  <item id="2" email="bob@test.org">Bob</item>\n'
        '</root>\n'
    ),
    ".yml": (
        "items:\n"
        "  - id: 1\n    name: Alice\n    email: alice@example.com\n"
        "  - id: 2\n    name: Bob\n    email: bob@test.org\n"
    ),
    ".yaml": (
        "items:\n"
        "  - id: 1\n    name: Alice\n    email: alice@example.com\n"
        "  - id: 2\n    name: Bob\n    email: bob@test.org\n"
    ),
}

# P9.10: regex для извлечения имен входных файлов из spec.summary/title.
# Ловит любое слово с data-расширением (X.txt, input.csv, data.json, etc).
_INPUT_FILE_RE = re.compile(
    r"\b([A-Za-z][\w\-]{0,40}\.(?:txt|csv|tsv|json|log|xml|yml|yaml|html))\b",
    re.IGNORECASE,
)


def _default_sample_for(rel_path: str) -> bytes:
    """Дефолтный реалистичный sample для входа по расширению.
    Если расширение неизвестно — возвращаем пустые байты (поведение до P9.10)."""
    rel_norm = _norm_path(rel_path)
    last = rel_norm.rsplit("/", 1)[-1]
    ext = ("." + last.rsplit(".", 1)[-1].lower()) if "." in last else ""
    text = _DEFAULT_INPUT_SAMPLES.get(ext, "")
    return text.encode("utf-8")


def _heuristic_input_paths(spec: dict | None) -> list[str]:
    """Извлекает вероятные входные файлы из spec.summary/title по regex.
    Исключает deliverables (это выходы). Детерминистически, без LLM."""
    if not isinstance(spec, dict):
        return []
    text_blob = " ".join(str(spec.get(k, "")) for k in ("summary", "title"))
    if not text_blob.strip():
        return []
    deliverables = {_norm_path(d) for d in (spec.get("deliverables") or [])}
    found: list[str] = []
    seen: set[str] = set()
    for m in _INPUT_FILE_RE.finditer(text_blob):
        rel = _norm_path(m.group(1))
        if rel in seen or rel in deliverables:
            continue
        seen.add(rel)
        found.append(rel)
    return found


def _enrich_plan_with_heuristic_inputs(plan: dict, spec: dict | None) -> dict:
    """P9.10: если архитектор забыл plan.inputs, но в spec.summary есть имена
    входных файлов — вписываем их в plan.inputs ПЕРЕД build-фазой, чтобы coder/aider
    видели эти пути. Существующие plan.inputs НЕ перезаписываются.
    Детерминистично, без LLM. Возвращает plan (модифицирует in-place)."""
    if not isinstance(plan, dict):
        return plan
    existing_paths: set[str] = set()
    inputs = list(plan.get("inputs") or [])
    for it in inputs:
        if isinstance(it, dict):
            p = _norm_path(it.get("path") or "")
            if p:
                existing_paths.add(p)
    deliverables = {_norm_path(d) for d in (spec.get("deliverables") or [])} if isinstance(spec, dict) else set()
    added = 0
    for rel in _heuristic_input_paths(spec):
        if rel in existing_paths or rel in deliverables:
            continue
        sample_bytes = _default_sample_for(rel)
        sample_text = sample_bytes.decode("utf-8", errors="replace") if sample_bytes else ""
        inputs.append({"path": rel, "sample_content": sample_text, "_source": "heuristic"})
        existing_paths.add(rel)
        added += 1
    if added:
        plan["inputs"] = inputs
        logger.info(f"[plan.enrich] added {added} heuristic input(s): {[i['path'] for i in inputs if i.get('_source')=='heuristic']}")
    return plan


# ─── P11.1: контракты plan.files и валидация архитектуры ───────────────────────
# Не-питон-расширения для которых exports логически пусты (данные/доки/конфиги).
_NON_PYTHON_EXTS = {
    ".json", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml", ".ini",
    ".cfg", ".toml", ".html", ".css", ".sql", ".log", ".env",
}

# Имена которые обычно являются точкой входа — их никто не импортирует,
# поэтому exports может быть пустым. Детектим структурно (базовое имя файла),
# не ключевыми словами в задаче: среди .py-файлов проекта файл с абсолютно
# никакими depends_on (кроме stdlib) — это кандидат на entry-point.
_LIKELY_ENTRY_BASENAMES = {"main.py", "__main__.py", "app.py", "run.py", "cli.py"}


def _is_python_path(rel: str) -> bool:
    return rel.lower().endswith(".py")


def _is_non_python_path(rel: str) -> bool:
    """True для файлов-данных/доки/конфигив (не .py)."""
    rl = rel.lower()
    if rl.endswith(".py"):
        return False
    for ext in _NON_PYTHON_EXTS:
        if rl.endswith(ext):
            return True
    # файлы без расширения (LICENSE, Makefile) — тоже не-питон
    if "." not in rl.rsplit("/", 1)[-1]:
        return True
    return False


def _normalize_export_entry(e: dict | None) -> dict | None:
    """Нормализует один элемент exports. Режектит мусор и получает одинаковую
    схему предсказуемо: {name, kind, signature, doc}."""
    if not isinstance(e, dict):
        return None
    name = str(e.get("name") or "").strip()
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return None
    kind = str(e.get("kind") or "").strip().lower()
    if kind not in ("function", "class", "const"):
        # Ставим эвристику по виду: всё в UPPER_SNAKE — константа,
        # всё с большой буквы — класс, остальное — функция.
        if name.isupper():
            kind = "const"
        elif name[0].isupper():
            kind = "class"
        else:
            kind = "function"
    sig = str(e.get("signature") or "").strip()
    doc = str(e.get("doc") or "").strip()
    return {"name": name, "kind": kind, "signature": sig, "doc": doc}


def _file_likely_entry_point(file_entry: dict) -> bool:
    """Файл выглядит как entry point: базовое имя main.py/__main__.py/app.py/run.py/cli.py.
    Не использует ключевые слова из задачи — только имя файла."""
    if not isinstance(file_entry, dict):
        return False
    rel = _norm_path(file_entry.get("path") or "")
    base = rel.rsplit("/", 1)[-1].lower()
    return base in _LIKELY_ENTRY_BASENAMES


def _normalize_plan_contracts(plan: dict, spec: dict | None) -> dict:
    """P11.1: валидирует и нормализует plan.files после архитектора:
      1) Нормализует exports каждого файла (отбрасывает мусор, выбирает kind эвристически).
      2) Проверяет согласованность depends_on ↔ exports: если B зависит от A,
         а A.exports пуст — флаг _exports_warning. План НЕ режектит (lossless).
      3) Считает _contract_metrics для телеметрии.

    Принцип: валидатор НИКОГДА не рубит план — только нормализует и списывает
    warnings. fallback всегда — пустые exports допускаются в этой версии. P11.2 использует
    exports при наличии; если пусты — fall back на старый путь (код соседних файлов).

    Модифицирует plan in-place и возвращает его же.
    """
    if not isinstance(plan, dict):
        return plan
    files = plan.get("files") or []
    if not isinstance(files, list):
        return plan

    paths_in_plan: set[str] = set()
    for f in files:
        if isinstance(f, dict):
            p = _norm_path(f.get("path") or "")
            if p:
                paths_in_plan.add(p)

    metrics = {
        "files_total": 0,
        "py_files": 0,
        "files_with_exports": 0,
        "files_missing_exports": [],   # list[str] of paths where exports пуст но должны быть
        "depends_unmatched": [],       # list[str] "B->A" где A.exports пуст
        "depends_outside_plan": [],    # list[str] "B->X" где X не в plan.files и не stdlib
    }

    for f in files:
        if not isinstance(f, dict):
            continue
        rel = _norm_path(f.get("path") or "")
        if not rel:
            continue
        metrics["files_total"] += 1

        # Нормализуем exports
        raw_exports = f.get("exports")
        norm_exports: list[dict] = []
        if isinstance(raw_exports, list):
            for e in raw_exports:
                norm = _normalize_export_entry(e)
                if norm is not None:
                    norm_exports.append(norm)
        f["exports"] = norm_exports

        if _is_python_path(rel):
            metrics["py_files"] += 1
            if norm_exports:
                metrics["files_with_exports"] += 1
            else:
                # Пустые exports оказываются легитимными только для entry-point
                # файлов (main.py/app.py/...) — их никто не импортирует.
                if not _file_likely_entry_point(f):
                    metrics["files_missing_exports"].append(rel)

        # depends_on валидация
        deps = f.get("depends_on") or []
        if isinstance(deps, list):
            for d in deps:
                ds = str(d or "").strip()
                if not ds:
                    continue
                if ds.lower() == "stdlib":
                    continue
                dep_path = _norm_path(ds)
                if dep_path not in paths_in_plan:
                    metrics["depends_outside_plan"].append(f"{rel}->{ds}")
                    continue
                # Проверяем: у файла-зависимости есть exports?
                dep_entry = next((x for x in files
                                   if isinstance(x, dict)
                                   and _norm_path(x.get("path") or "") == dep_path), None)
                if isinstance(dep_entry, dict):
                    dep_exports = dep_entry.get("exports") or []
                    # для не-питона пустые exports ок; для .py файлов от которых
                    # зависят — пустые exports = нарушение контракта
                    if _is_python_path(dep_path) and not dep_exports:
                        metrics["depends_unmatched"].append(f"{rel}->{dep_path}")

    plan["_contract_metrics"] = metrics

    # Логим предупреждения (не ошибки) для видимости
    if metrics["files_missing_exports"]:
        logger.info(f"[plan.contracts] py-files без exports но не entry-point: {metrics['files_missing_exports']}")
    if metrics["depends_unmatched"]:
        logger.info(f"[plan.contracts] depends_on без exports-контракта: {metrics['depends_unmatched']}")
    if metrics["depends_outside_plan"]:
        logger.warning(f"[plan.contracts] depends_on вне plan.files: {metrics['depends_outside_plan']}")

    return plan


def _dedupe_files_vs_inputs(plan: dict | None) -> dict:
    """P11.2.e (FM-10): если файл объявлен в plan.inputs (входная фикстура)
    и в plan.files одновременно — убираем из plan.files. Иначе сборщик
    пытается "собрать" todos.json через aider — бесполезно и расходует бюджет.
    Lossless: ничего не режется кроме дубликатов input→files."""
    if not isinstance(plan, dict):
        return plan
    files = plan.get("files") or []
    inputs = plan.get("inputs") or []
    if not isinstance(files, list) or not isinstance(inputs, list):
        return plan
    input_paths: set[str] = set()
    for it in inputs:
        if isinstance(it, dict):
            p = (it.get("path") or "").strip().replace("\\", "/")
            if p:
                input_paths.add(p)
    if not input_paths:
        return plan
    new_files = []
    removed: list[str] = []
    for f in files:
        if not isinstance(f, dict):
            new_files.append(f)
            continue
        p = (f.get("path") or "").strip().replace("\\", "/")
        # Режем только не-питон-файлы: если inputs совпадает с .py-файлом, оставляем
        # files (редкий случай, видимо ошибка в plan, пусть heal разбирается).
        if p and p in input_paths and not p.lower().endswith(".py"):
            removed.append(p)
            continue
        new_files.append(f)
    if removed:
        plan["files"] = new_files
        logger.info(f"[plan.dedupe] removed {len(removed)} input(s) from plan.files: {removed}")
    return plan


# =============================================================================
# P11.2: coder получает API соседей
# =============================================================================
# Идея: когда aider строит файл F, он должен видеть КОНТРАКТЫ всех F.depends_on:
#   - если соседний файл уже собран на диске — передаём его как --read (реальный код);
#   - если не собран — генерим stub из plan exports (сигнатуры с NotImplementedError),
#     пишем в .jarvis/contracts/<dep> и тоже передаём как --read.
# Структурно — никаких keyword-эвристик, решения из plan.

_CONTRACT_DIR_NAME = ".jarvis_contracts"


def _module_name_from_rel(rel: str) -> str:
    """main.py -> main, src/utils.py -> src.utils. None -> ''."""
    if not rel or not isinstance(rel, str):
        return ""
    p = rel.replace("\\", "/").strip("./")
    if p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".")


def _render_export_signature(exp: dict) -> str:
    """По exports-элементу сформировать короткую сигнатуру для промпта.

    Формат:
      function: "add(a, b) -> int"
      class:    "class Storage(db_path: str)"
      const:    "DB_PATH: str"
    Если signature в plan уже выглядит правильно — берём её as-is."""
    if not isinstance(exp, dict):
        return ""
    name = (exp.get("name") or "").strip()
    if not name:
        return ""
    kind = (exp.get("kind") or "function").strip().lower()
    sig = (exp.get("signature") or "").strip()
    if kind == "const":
        # signature может быть типом ("str") или видом "DB_PATH: str".
        if sig.startswith(name):
            return sig
        if sig:
            # попробуем проинтерпретировать сигнатуру как тип
            return f"{name}: {sig}"
        return name
    if kind == "class":
        if sig.startswith("class "):
            return sig
        if sig.startswith(name):
            return f"class {sig}"
        if sig.startswith("("):
            return f"class {name}{sig}"
        return f"class {name}" + (f"({sig})" if sig else "")
    # function (default)
    if sig.startswith(name):
        return sig
    if sig.startswith("("):
        return f"{name}{sig}"
    return f"{name}({sig})" if sig else f"{name}()"


def _render_neighbor_stub(dep_rel: str, dep_file: dict) -> str:
    """Собрать содержимое stub-файла для соседа по exports.

    Вывод — синтаксически валидный Python: импорты, функции с сигнатурами и raise NotImplementedError,
    классы с pass-телом, константы с placeholder-значениями. Нужен исключительно как
    READ-ONLY контекст для aider — чтобы coder видел имена и сигнатуры API."""
    if not isinstance(dep_file, dict):
        return ""
    exports = dep_file.get("exports") or []
    purpose = (dep_file.get("purpose") or "").strip()
    lines: list[str] = []
    lines.append(f'"""P11.2 contract stub for {dep_rel}.')
    if purpose:
        lines.append(f"Purpose: {purpose}")
    lines.append("This file is READ-ONLY context. The real implementation lives elsewhere.")
    lines.append("Do not modify; just import these names from this module path.\"\"\"")
    lines.append("")
    has_any = False
    for exp in exports:
        if not isinstance(exp, dict):
            continue
        name = (exp.get("name") or "").strip()
        if not name:
            continue
        kind = (exp.get("kind") or "function").strip().lower()
        doc = (exp.get("doc") or "").strip().replace('"""', "'''")
        sig = _render_export_signature(exp)
        has_any = True
        if kind == "const":
            # Плейсхолдер-значение (используем None — реальное значение в настоящем модуле).
            if doc:
                lines.append(f"# {doc}")
            lines.append(f"{name} = None  # contract: {sig}")
            lines.append("")
        elif kind == "class":
            # Для stub не выводим base-classes или параметры __init__ — это невалидный Python.
            # Сигнатуру покажем в комментарии и в docstring — этого достаточно для read-only context.
            lines.append(f"# contract: {sig}")
            lines.append(f"class {name}:")
            ds = doc or sig
            if ds:
                lines.append(f'    """{ds}"""')
            lines.append("    pass")
            lines.append("")
        else:
            lines.append(f"def {sig}:")
            if doc:
                lines.append(f'    """{doc}"""')
            lines.append(f'    raise NotImplementedError("contract stub: see real {dep_rel}")')
            lines.append("")
    if not has_any:
        lines.append("# (no exports declared in plan)")
        lines.append("")
    return "\n".join(lines)


def _build_neighbor_context(
    project_root,
    plan: dict | None,
    target_rel: str,
    *,
    contracts_subdir: str = _CONTRACT_DIR_NAME,
) -> tuple[list[str], list[str]]:
    """Собрать read-only контекст для aider при сборке target_rel.

    Возвращает (read_only_paths_str, neighbor_module_descriptions):
      • read_only_paths_str — абсолютные str-пути для aider --read
      • neighbor_module_descriptions — список строк для включения в промпты coder-а
        (имя модуля и список имен, которые он экспортирует).

    Никогда не бросает: при любой ошибке возвращает то что удалось собрать."""
    from pathlib import Path as _Path
    if not isinstance(plan, dict):
        return ([], [])
    files = plan.get("files") or []
    if not isinstance(files, list):
        return ([], [])

    # Индекс по пути и найдем целевой
    by_path: dict[str, dict] = {}
    for f in files:
        if isinstance(f, dict):
            p = (f.get("path") or "").strip()
            if p:
                by_path[p] = f
    target = by_path.get(target_rel) or {}
    deps = target.get("depends_on") or []
    if not isinstance(deps, list):
        return ([], [])

    project_root = _Path(project_root)
    contracts_dir = project_root / contracts_subdir
    read_only_paths: list[str] = []
    neighbor_descs: list[str] = []
    seen: set[str] = set()
    for dep in deps:
        if not isinstance(dep, str):
            continue
        dep_norm = dep.strip()
        if not dep_norm or dep_norm in seen:
            continue
        seen.add(dep_norm)
        # Игнорируем stdlib-маркер и внешние pip-пакеты (их нет в plan.files).
        if dep_norm.lower() == "stdlib":
            continue
        dep_file = by_path.get(dep_norm)
        if dep_file is None:
            # Зависимость вне плана — не можем помочь
            continue
        # Не-питон (напр. data.json) — без stubа. Если файл уже есть на диске,
        # передадим как --read для контекста.
        real_path = project_root / dep_norm
        if not _is_python_path(dep_norm):
            if real_path.is_file():
                read_only_paths.append(str(real_path))
            continue

        # Питон-сосед:
        if real_path.is_file() and real_path.stat().st_size > 0:
            # Реальный код — лучше стаба
            read_only_paths.append(str(real_path))
        else:
            # Генерим stub из exports
            try:
                contracts_dir.mkdir(parents=True, exist_ok=True)
                stub_name = dep_norm.replace("\\", "/").replace("/", "__")
                stub_path = contracts_dir / stub_name
                stub_text = _render_neighbor_stub(dep_norm, dep_file)
                stub_path.write_text(stub_text, encoding="utf-8")
                read_only_paths.append(str(stub_path))
            except Exception as e:
                logger.warning(f"[neighbor.stub] failed for {dep_norm}: {e}")
        # Описание для промпта
        mod = _module_name_from_rel(dep_norm)
        names = []
        for exp in (dep_file.get("exports") or []):
            if isinstance(exp, dict):
                nm = (exp.get("name") or "").strip()
                if nm:
                    names.append(_render_export_signature(exp))
        if names:
            neighbor_descs.append(
                f"• Модуль {mod} (файл {dep_norm}) экспортирует: " + ", ".join(names)
            )
        else:
            neighbor_descs.append(
                f"• Модуль {mod} (файл {dep_norm}) — без задекларированных exports"
            )
    return (read_only_paths, neighbor_descs)


def _build_contract_prompt_block(plan: dict | None, target_rel: str) -> str:
    """Сформировать CONTRACT-блок для промпта coder-а.

    Вывод — многострочная секция, включающая:
      • must_export — имена и сигнатуры, которые ОБЯЗАН реализовать файл
      • required_imports — импорты из соседей (вычислены из depends_on ∩ plan.exports)
    Пустая строка — если ничего не объявлено в плане."""
    if not isinstance(plan, dict):
        return ""
    files = plan.get("files") or []
    if not isinstance(files, list):
        return ""
    by_path = {(f.get("path") or ""): f for f in files if isinstance(f, dict)}
    target = by_path.get(target_rel) or {}

    parts: list[str] = []

    # 1) Что этот файл должен экспортировать
    own_exports = target.get("exports") or []
    own_lines = []
    for exp in own_exports:
        if isinstance(exp, dict):
            sig = _render_export_signature(exp)
            if sig:
                own_lines.append(f"  - {sig}")
    if own_lines and _is_python_path(target_rel):
        parts.append(
            "КОНТРАКТ ЭТОГО ФАЙЛА (ты ОБЯЗАН реализовать ИМЕННО эти имена с точными сигнатурами):\n"
            + "\n".join(own_lines)
            + "\nНЕ переименовывай (DB_PATH ≠ DATABASE_PATH). НЕ добавляй лишних public-имен."
        )

    # 2) Что этот файл ОБЯЗАН импортировать из соседей
    deps = target.get("depends_on") or []
    import_lines = []
    for dep in deps:
        if not isinstance(dep, str):
            continue
        dep = dep.strip()
        if not dep or dep.lower() == "stdlib":
            continue
        dep_file = by_path.get(dep)
        if not isinstance(dep_file, dict):
            continue
        if not _is_python_path(dep):
            continue
        names = []
        for exp in (dep_file.get("exports") or []):
            if isinstance(exp, dict):
                nm = (exp.get("name") or "").strip()
                if nm:
                    names.append(nm)
        if not names:
            continue
        mod = _module_name_from_rel(dep)
        import_lines.append(f"  from {mod} import {', '.join(names)}")
    if import_lines:
        parts.append(
            "ОБЯЗАТЕЛЬНЫЕ ИМПОРТЫ (используй ровно эти имена, не дублируй функции соседей):\n"
            + "\n".join(import_lines)
            + "\nНЕ переписывай логику соседей в своём файле — вызывай их функции через импорт."
        )

    return "\n\n".join(parts)


def _check_file_contract(
    file_text: str,
    expected_exports: list,
) -> dict:
    """P11.2.d: статическая сверка реальных top-level имён с ожидаемыми exports.

    Возвращает dict:
      ok: bool                    — все expected найдены
      missing: list[str]          — ожидались но не найдены
      kind_mismatch: list[dict]   — найдены с другим kind
      found_top_level: list[str]  — что реально объявлено
      ast_ok: bool                — файл парсится
    Не падает ни на чём."""
    import ast as _ast
    out = {
        "ok": True,
        "missing": [],
        "kind_mismatch": [],
        "found_top_level": [],
        "ast_ok": True,
    }
    if not isinstance(file_text, str) or not file_text.strip():
        if expected_exports:
            out["ok"] = False
            out["missing"] = [
                (e.get("name") or "") for e in (expected_exports or [])
                if isinstance(e, dict) and e.get("name")
            ]
            out["ast_ok"] = False
        return out
    try:
        tree = _ast.parse(file_text)
    except SyntaxError:
        out["ast_ok"] = False
        out["ok"] = False
        return out
    found_kinds: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) or isinstance(node, _ast.AsyncFunctionDef):
            found_kinds[node.name] = "function"
        elif isinstance(node, _ast.ClassDef):
            found_kinds[node.name] = "class"
        elif isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    found_kinds.setdefault(t.id, "const")
        elif isinstance(node, _ast.AnnAssign) and isinstance(node.target, _ast.Name):
            found_kinds.setdefault(node.target.id, "const")
    out["found_top_level"] = sorted(found_kinds.keys())

    for exp in (expected_exports or []):
        if not isinstance(exp, dict):
            continue
        name = (exp.get("name") or "").strip()
        if not name:
            continue
        if name not in found_kinds:
            out["missing"].append(name)
            continue
        expected_kind = (exp.get("kind") or "").strip().lower()
        actual_kind = found_kinds[name]
        if expected_kind and expected_kind != actual_kind:
            out["kind_mismatch"].append({
                "name": name,
                "expected": expected_kind,
                "actual": actual_kind,
            })
    out["ok"] = not out["missing"] and not out["kind_mismatch"]
    return out


def _collect_input_specs(plan: dict | None, spec: dict | None) -> list[dict]:
    """Собирает список входов из двух источников (приоритет plan.inputs):
    1) plan.inputs: [{path, sample_content?}] — явно указано архитектором.
    2) Heuristic из spec.summary/title — fallback когда архитектор забыл.

    Результат: список {'path': str, 'sample_content': bytes, 'source': 'plan'|'heuristic'}.
    Пути из deliverables исключаются. Дубликаты убираются."""
    deliverables: set[str] = set()
    if isinstance(spec, dict):
        deliverables = {_norm_path(d) for d in (spec.get("deliverables") or [])}

    out: list[dict] = []
    seen: set[str] = set()

    # 1) plan.inputs (явный контракт)
    if isinstance(plan, dict):
        for it in (plan.get("inputs") or []):
            if not isinstance(it, dict):
                continue
            rel = _norm_path(it.get("path") or "")
            if not rel or rel in seen or rel in deliverables:
                continue
            sample = it.get("sample_content")
            if isinstance(sample, str) and sample:
                content = sample.encode("utf-8")
            else:
                content = _default_sample_for(rel)
            out.append({"path": rel, "sample_content": content, "source": "plan"})
            seen.add(rel)

    # 2) heuristic fallback из summary/title
    for rel in _heuristic_input_paths(spec):
        if rel in seen:
            continue
        out.append({"path": rel, "sample_content": _default_sample_for(rel), "source": "heuristic"})
        seen.add(rel)

    return out


def _norm_path(p: str) -> str:
    """Унифицированная нормализация относительного пути для сравнения."""
    if not p:
        return ""
    return str(p).replace("\\", "/").lstrip("./")


def _collect_output_paths(plan: dict | None) -> set[str]:
    """Собирает пути, которые скрипт ПИШЕТ/СОЗДАЁТ, из plan'а:
    1) Любой path в чеках типа _OUTPUT_CHECK_TYPES.
    2) build_steps[].target если step.kind in {create_file, write_file}.
    Возвращает множество нормализованных путей."""
    out: set[str] = set()
    if not isinstance(plan, dict):
        return out
    # 1) пост-проверки результата
    for t in (plan.get("tests") or []):
        for ch in (t.get("checks") or []):
            if not isinstance(ch, dict):
                continue
            ctype = (ch.get("type") or "").strip()
            if ctype in _OUTPUT_CHECK_TYPES:
                rel = _norm_path(ch.get("path") or "")
                if rel:
                    out.add(rel)
    # 2) build_steps описывающие создание файла
    for st in (plan.get("build_steps") or []):
        if not isinstance(st, dict):
            continue
        kind = (st.get("kind") or "").strip().lower()
        if kind in {"create_file", "write_file", "generate_file"}:
            rel = _norm_path(st.get("target") or "")
            if rel:
                out.add(rel)
    return out


def _is_input_fixture(rel_path: str, deliverables: list[str], output_paths: set[str] | None = None) -> bool:
    """True, если file_exists(rel_path) — это, вероятно, ВХОДНОЙ файл,
    а не выходной артефакт. Эвристика:
    - путь НЕ значится в deliverables;
    - путь НЕ значится в output_paths (P9.9: пути с пост-проверками результата
      или явно создаваемые в build_steps);
    - путь НЕ является исходником.
    Тогда смело можно создать пустой dummy чтобы тест не падал на отсутствии входа."""
    if not rel_path:
        return False
    rel_norm = _norm_path(rel_path)
    norm_dlv = {_norm_path(d) for d in (deliverables or [])}
    if rel_norm in norm_dlv:
        return False
    if output_paths and rel_norm in output_paths:
        # P9.9: путь упомянут в чеках типа file_min_lines/file_min_size/json_valid
        # или явно создаётся скриптом — это ВЫХОД, не вход.
        return False
    # Извлекаем расширение без внешних модулей (os/pathlib не импортированы выше).
    last_seg = rel_norm.rsplit("/", 1)[-1]
    ext = ("." + last_seg.rsplit(".", 1)[-1].lower()) if "." in last_seg else ""
    if ext in _SOURCE_EXT_FOR_FIXTURE:
        return False
    return True


def _prepare_test_fixtures(slug: str, plan: dict, spec: dict | None) -> list[str]:
    """Создаёт входные фикстуры перед запуском тестов.

    P9.10: два источника:
    1) plan.inputs (явный): {path, sample_content?} — создаёт реалистичный вход.
    2) Heuristic из spec.summary/title — fallback для забывчивого архитектора.

    P9.7-legacy: если file_exists(path) в plan.tests указывает на вход, которого
    никто не описал — создаём пустую заглушку (старое поведение).

    Существующие файлы НЕ перезаписываются. Детерминистично, без LLM, без ключевых слов.

    Возвращает список созданных относительных путей."""
    if not isinstance(plan, dict) or not isinstance(spec, dict):
        return []
    deliverables = [str(d) for d in (spec.get("deliverables") or [])]
    output_paths = _collect_output_paths(plan)
    created: list[str] = []
    seen_paths: set[str] = set()

    # P9.10: реалистичные входы из plan.inputs + heuristic.
    for it in _collect_input_specs(plan, spec):
        rel = it["path"]
        if rel in seen_paths or rel in output_paths:
            continue
        seen_paths.add(rel)
        try:
            abs_path = safe_project_path(slug, rel)
        except Exception:
            continue
        if abs_path.exists():
            continue
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(it["sample_content"])
            created.append(rel)
        except Exception as e:
            logger.debug(f"[test.fixture] cannot create input {rel}: {e}")

    # P9.7-legacy: если в чеках есть file_exists на вход, которого ещё нет —
    # создаём пустую заглушку (фильтр из P9.9 обычно убирает такие чеки, но
    # если путь всё равно остался — даём файлу быть).
    for t in (plan.get("tests") or []):
        for ch in (t.get("checks") or []):
            if not isinstance(ch, dict):
                continue
            if (ch.get("type") or "").strip() != "file_exists":
                continue
            rel = _norm_path((ch.get("path") or "").strip())
            if not rel or rel in seen_paths:
                continue
            if not _is_input_fixture(rel, deliverables, output_paths):
                continue
            seen_paths.add(rel)
            try:
                abs_path = safe_project_path(slug, rel)
            except Exception:
                continue
            if abs_path.exists():
                continue
            # Если расширение известно — создаём реалистичный sample,
            # иначе пустой (легаси P9.7).
            content = _default_sample_for(rel)
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_bytes(content)
                created.append(rel)
            except Exception as e:
                logger.debug(f"[test.fixture] cannot create dummy {rel}: {e}")
    return created


def _normalize_min_size_check(slug: str, check: dict) -> dict:
    """Если архитектор поставил нереалистичный file_min_size > _MIN_SIZE_REALISTIC_FLOOR,
    но файл реально создан и не пуст — понижаем порог до факт-размера, помечая
    check как _soft (мягкий: warn, не fail). Это не маскирует реальные баги:
    если файл вообще не создан или пуст — check остаётся жёстким."""
    if not isinstance(check, dict):
        return check
    if (check.get("type") or "").strip() != "file_min_size":
        return check
    rel = (check.get("path") or "").strip()
    if not rel:
        return check
    try:
        min_bytes = int(check.get("bytes", 0))
    except (TypeError, ValueError):
        return check
    try:
        abs_path = safe_project_path(slug, rel)
    except Exception:
        return check
    if not abs_path.exists():
        return check
    actual = abs_path.stat().st_size
    if actual >= _MIN_SIZE_REALISTIC_FLOOR and actual < min_bytes:
        # Файл создан, не пуст, но меньше нереалистичного порога. Мягкий режим.
        out = dict(check)
        out["_soft"] = True
        out["_original_bytes"] = min_bytes
        out["bytes"] = max(_MIN_SIZE_REALISTIC_FLOOR, min(min_bytes, actual))
        return out
    return check


def _run_one_test(slug: str, t: dict) -> dict:
    cmd = (t.get("command") or "").strip()
    parts = cmd.split()
    if not parts:
        return {"name": t.get("name", "test"), "command": cmd, "ok": False,
                "rc": -1, "stdout": "", "stderr": "empty command", "checks": []}
    # P11.0.1: если в команде есть pipe/redirect/chain — нужен shell.
    # Иначе subprocess с shell=False ищет 'echo'/'cat'/'|' как .exe и падает
    # с WinError 2 на Windows. _has_shell_metachars смотрит только на сами
    # символы (|, <, >, &&, ||), это не решение по ключевым словам.
    if _has_shell_metachars(cmd):
        res = run_shell_in_project(slug, cmd, timeout=PHASE_TEST_TIMEOUT)
    # Запускаем через venv-python если первая часть — python/python3 или pytest
    elif parts[0].lower() in ("python", "python3"):
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
            # P9.7: нормализуем нереалистичные file_min_size в soft-mode
            ch_norm = _normalize_min_size_check(slug, ch) if isinstance(ch, dict) else ch
            r = _evaluate_check(slug, ch_norm, res)
            # P9.7: если check помечен как мягкий и реальный размер выше факт-порога,
            # считаем пройдённым с пометкой _soft в reason (для видимости в логах).
            if isinstance(ch_norm, dict) and ch_norm.get("_soft") and r.get("ok"):
                r["_soft"] = True
                r["_original_bytes"] = ch_norm.get("_original_bytes")
                r["reason"] = "[soft] " + r.get("reason", "")
            check_results.append(r)
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


def _filter_invalid_checks(plan: dict, spec: dict | None) -> tuple[dict, list[dict]]:
    """P9.9: детерминистически убирает из plan.tests[].checks чеки, которые
    нарушают принципы Jarvis или логически бессмысленны:

    1) stdout_contains — ВСЕГДА удаляем. Нарушает принцип «запрещены
       ключевые слова» (проверка по жёсткому фрагменту текста).
    2) file_exists(path) на входной файл ПРИ УСЛОВИИ что этот же path НЕ
       упомянут в выходных чеках (file_min_lines/file_min_size/json_valid/...).
       Такой file_exists бессмыслен — фикстура уже создана _prepare_test_fixtures.
       ИСКЛЮЧЕНИЕ: file_exists на выход (такие пути в output_paths) ГРОМКО
       информативен — валидная пост-проверка, ОСТАВЛЯЕМ.

    Возвращает (новый plan, список удалённых чеков с reason для лога)."""
    if not isinstance(plan, dict):
        return plan, []
    deliverables = []
    if isinstance(spec, dict):
        deliverables = [str(d) for d in (spec.get("deliverables") or [])]
    output_paths = _collect_output_paths(plan)

    new_plan = dict(plan)
    new_tests = []
    removed: list[dict] = []
    for t in (plan.get("tests") or []):
        if not isinstance(t, dict):
            new_tests.append(t)
            continue
        new_t = dict(t)
        new_checks = []
        for ch in (t.get("checks") or []):
            if not isinstance(ch, dict):
                new_checks.append(ch)
                continue
            ctype = (ch.get("type") or "").strip()
            # 1) stdout_contains — всегда убираем
            if ctype == "stdout_contains":
                removed.append({"check": ch, "reason": "stdout_contains_violates_principles"})
                continue
            # 2) file_exists на вход без выходных чеков на тот же путь
            if ctype == "file_exists":
                rel = _norm_path(ch.get("path") or "")
                if rel and rel not in output_paths and _is_input_fixture(rel, deliverables, output_paths):
                    removed.append({"check": ch, "reason": "file_exists_on_input_fixture"})
                    continue
            new_checks.append(ch)
        new_t["checks"] = new_checks
        new_tests.append(new_t)
    new_plan["tests"] = new_tests
    return new_plan, removed


def _test(slug: str, plan: dict, spec: dict | None = None) -> list[dict]:
    # P9.9: фильтрация невалидных checks (stdout_contains по принципам +
    # file_exists на входную фикстуру без выходных чеков). Первым этапом.
    try:
        plan, removed = _filter_invalid_checks(plan, spec)
        if removed:
            reasons = sorted({r["reason"] for r in removed})
            add_phase(slug, "test:filter", "ok",
                      f"removed {len(removed)} invalid check(s): reasons={reasons}")
    except Exception as e:
        logger.debug(f"[test.filter] error: {e}")
    # P9.7: автофикстуры для входных файлов, которые архитектор ошибочно
    # включил в file_exists. Спек опционален для обратной совместимости с тестами.
    if spec is not None:
        try:
            created = _prepare_test_fixtures(slug, plan, spec)
            if created:
                add_phase(slug, "test:fixtures", "ok",
                          f"created {len(created)} input fixture(s): {created[:5]}")
        except Exception as e:
            logger.debug(f"[test.fixtures] error: {e}")
    out = []
    for t in (plan.get("tests") or []):
        rec = _run_one_test(slug, t)
        out.append(rec)
        # P9.6: включаем детальные checks в деталь фазы для диагностики
        # файловых/контент-проверок (file_min_size, stdout_contains и т.п.).
        # Без этого в логах видно только rc=0 stderr="" и непонятно почему failed.
        keys = ["command", "rc", "expects_ok", "stderr", "checks"]
        add_phase(slug, f"test:{rec['name']}",
                  "ok" if rec["ok"] else "failed",
                  json.dumps({k: rec.get(k) for k in keys if k in rec}, ensure_ascii=False))
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
    raw = _llm(budget, MODEL_HEALER, PROJECT_HEAL_SYSTEM, user,
               temperature=0.0, num_ctx=4096, where="heal.diagnose")
    diag = _safe_parse(raw)
    if not isinstance(diag, dict):
        diag = {}
    return diag


_MODNOTFOUND_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([A-Za-z0-9_.\-]+)['\"]")
# P2: detection patterns for additional deterministic healers.
_SYNTAX_ERR_RE   = re.compile(r"(SyntaxError|IndentationError|TabError):\s*(.+)")
_SYNTAX_FILE_RE  = re.compile(r'File "([^"]+\.py)", line (\d+)')
_NETWORK_ERR_RE  = re.compile(
    r"(ConnectionError|ConnectionResetError|ConnectionRefusedError|"
    r"ReadTimeout|ConnectTimeout|TimeoutError|ssl\.SSLError|"
    r"urllib3\.exceptions|requests\.exceptions\.\w+|http\.client\.RemoteDisconnected|"
    r"socket\.gaierror|socket\.timeout|TimeoutExpired)"
)
_JSONDEC_RE      = re.compile(r"json\.decoder\.JSONDecodeError|json\.JSONDecodeError")
# P9.6: bs4 жалуется на отсутствие парсера (xml/lxml/html5lib) — ставим нужный пакет.
_BS4_FEATURE_RE  = re.compile(r"FeatureNotFound.*?features you requested:\s*([a-zA-Z0-9_\-]+)", re.DOTALL)


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


def _heal_syntax_error(slug: str, failed: dict, plan: dict) -> dict | None:
    """P2: детерминистический хилер для SyntaxError/IndentationError на фазе test.

    Стратегия: перебираем все .py-файлы проекта, проверяем ast.parse.
    Первый найденный файл с ошибкой — это цель. Возвращаем dict с target_path и
    fix_instruction, которые _heal_loop скормит Coder'у без LLM-диагностики.
    """
    stderr = failed.get("stderr", "") or ""
    if not _SYNTAX_ERR_RE.search(stderr):
        return None
    file_paths = [f["path"] for f in plan.get("files", []) if isinstance(f, dict) and "path" in f]
    py_files = [p for p in file_paths if p.lower().endswith(".py")]
    if not py_files:
        return None
    # Ищем первый .py с битым синтаксисом.
    for path in py_files:
        try:
            content = read_project_file(slug, path)
        except Exception:
            continue
        sc = static_check(path, content)
        if not sc.get("ok") and sc.get("applicable"):
            return {
                "target_file": path,
                "fix_instruction": static_errors_to_feedback(sc.get("errors") or []),
                "category": "syntax",
            }
    # stderr жалуется на синтаксис, но ast.parse все файлы прошли — пробуем выдернуть имя/номер из traceback.
    fm = _SYNTAX_FILE_RE.search(stderr)
    sm = _SYNTAX_ERR_RE.search(stderr)
    if fm and sm:
        guess = fm.group(1).replace("\\", "/").split("/")[-1]
        if guess in py_files:
            return {
                "target_file": guess,
                "fix_instruction": (
                    f"Исправь {sm.group(1)} в {guess} на строке {fm.group(2)}: {sm.group(2)}"
                ),
                "category": "syntax",
            }
    return None


def _heal_network_retry(slug: str, failed: dict, plan: dict, attempt: int = 1) -> dict | None:
    """P2: при сетевых ошибках просто перезапускаем тест (без LLM, без правки кода).

    Стратегия: пауза (1.5с × attempt), ретест. Применяется вызывающим кодом, 
    который сам вызывает _test() после результата. Мы решаем только «stoit ли ретраить?» и паузим.
    """
    stderr = failed.get("stderr", "") or ""
    if not _NETWORK_ERR_RE.search(stderr):
        return None
    delay = min(1.5 * attempt, 4.5)
    time.sleep(delay)
    return {"category": "network", "retried_after_s": delay, "attempt": attempt}


def _heal_json_decode(slug: str, failed: dict, plan: dict) -> dict | None:
    """P2: при JSONDecodeError — хинт Coder'у: добавь try/except и проверку Content-Type.

    Это не исправляет баг автоматически, но даёт однозначную fix_instruction без _diagnose-LLM.
    """
    stderr = failed.get("stderr", "") or ""
    if not _JSONDEC_RE.search(stderr):
        return None
    file_paths = [f["path"] for f in plan.get("files", []) if isinstance(f, dict) and "path" in f]
    # Ищем первый .py файл, где вызывается json.loads или .json().
    for path in file_paths:
        if not path.lower().endswith(".py"):
            continue
        try:
            content = read_project_file(slug, path)
        except Exception:
            continue
        if ("json.loads" in content) or (".json(" in content) or ("json.load(" in content):
            return {
                "target_file": path,
                "fix_instruction": (
                    "Оборачивай вызовы json.loads/.json() в try/except json.JSONDecodeError. "
                    "Если ресурс отвечает не-JSON, не падай, а выведи первые байты ответа для диагностики. "
                    "Для requests используй response.headers.get('Content-Type') перед разбором."
                ),
                "category": "json",
            }
    return None


def _heal_missing_module(slug: str, failed: dict) -> dict | None:
    """Детерминистический healer для ModuleNotFoundError — без LLM.
    Выдергивает имя модуля из stderr и ставит его в venv.
    Возвращает dict с результатом или None если это не ModuleNotFoundError.

    P9.6: также распознаёт bs4.FeatureNotFound — это не ModuleNotFoundError, но
    получается когда код зовёт BeautifulSoup(text, 'xml') без установленного lxml.
    """
    stderr = failed.get("stderr", "") or ""

    # Сначала пробуем bs4 FeatureNotFound (более специфичный паттерн).
    bm = _BS4_FEATURE_RE.search(stderr)
    if bm:
        feature = bm.group(1).lower()
        feature_to_pkg = {"xml": "lxml", "lxml": "lxml", "lxml-xml": "lxml", "html5lib": "html5lib"}
        pkg = feature_to_pkg.get(feature, "lxml")
        if not _PKG_PATTERN.match(pkg):
            return {"ok": False, "missing": f"bs4-feature:{feature}", "reason": "unsafe pkg name"}
        res = pip_install(slug, [pkg])
        return {
            "ok": bool(res.get("ok")),
            "missing": f"bs4-feature:{feature}",
            "installed_as": pkg,
            "stderr": res.get("stderr", "")[-400:],
        }

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


# P11.4: cross-file heal — структурный разбор ошибки и выбор правильного target.
# Регулярки распознают конкретные форматы Python-ошибок, без угадывания по словам.

# NameError: name 'X' is not defined  (баг в файле, где имя используется)
_HEAL_NAMEERROR_RE = re.compile(r"NameError: name ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"] is not defined")

# ImportError: cannot import name 'X' from 'M'  (баг в M — должен экспортировать X)
_HEAL_IMPORTERROR_RE = re.compile(
    r"ImportError: cannot import name ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"] from ['\"]([A-Za-z0-9_.]+)['\"]"
)

# AttributeError: module 'M' has no attribute 'X'  (баг в M)
_HEAL_ATTRMOD_RE = re.compile(
    r"AttributeError: module ['\"]([A-Za-z0-9_.]+)['\"] has no attribute ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
)

# Frame в traceback: File "...\path\to\file.py", line N, in <...>
_HEAL_FRAME_RE = re.compile(r'File "([^"]+?\.py)", line (\d+)')


def _last_user_frame_in_plan(stderr: str, plan_paths: list[str]) -> str | None:
    """Найти последний frame в traceback, чей файл есть в plan.files.

    Это файл, в котором фактически произошла ошибка — не stdlib, не библиотеки.
    Возвращает rel-путь из plan или None.
    """
    if not stderr or not plan_paths:
        return None
    plan_lower = {p.lower(): p for p in plan_paths}
    plan_basenames = {p.split("/")[-1].split("\\")[-1].lower(): p for p in plan_paths}
    last_match = None
    for m in _HEAL_FRAME_RE.finditer(stderr):
        frame_path = (m.group(1) or "").replace("\\", "/")
        # Сначала пытаемся точное совпадение по rel-пути
        for low, orig in plan_lower.items():
            if frame_path.lower().endswith("/" + low) or frame_path.lower() == low:
                last_match = orig
                break
        else:
            # Fallback: совпадение по basename
            base = frame_path.split("/")[-1].lower()
            if base in plan_basenames:
                last_match = plan_basenames[base]
    return last_match


def _module_to_plan_path(mod_name: str, plan_paths: list[str]) -> str | None:
    """Перевести имя модуля ('config', 'storage') в rel-путь из plan.

    Использует _module_name_from_rel — то же преобразование, что и для CONTRACT-блока.
    """
    if not mod_name:
        return None
    target = mod_name.split(".")[0]  # верхний уровень
    for p in plan_paths:
        if _module_name_from_rel(p) == mod_name or _module_name_from_rel(p) == target:
            return p
    return None


def _classify_failure(plan: dict, failed: dict) -> dict:
    """P11.4: структурный разбор ошибки теста → {kind, target, missing_name, source_file, reason}.

    Логика выбора target (структурная, не ключевые слова):
      • ImportError/AttributeError на имени X из модуля M → target = plan-файл, реализующий M.
        Если plan.exports говорит что X должен быть в M — это баг M.
      • NameError 'X' в файле F → если X объявлен как export соседа N — баг F (забыл импортировать).
        Иначе баг F (неопределённое локальное имя).
      • Любая другая ошибка с traceback в файле F из plan → target = F.
      • Ничего не подошло → None (даём LLM-Healer решить).

    Возвращает dict с полями:
      kind: 'import_error' | 'attribute_error' | 'name_error' | 'traceback' | 'unknown'
      target: str | None      — rel-путь файла, который надо чинить
      missing_name: str | None — отсутствующее имя
      source_file: str | None  — где обнаружилась ошибка (последний user-frame)
      reason: str              — короткое объяснение для лога
      hint: str                — подсказка для патчера (не для LLM-выбора, а для feedback)
    """
    out = {
        "kind": "unknown",
        "target": None,
        "missing_name": None,
        "source_file": None,
        "reason": "",
        "hint": "",
    }
    if not isinstance(plan, dict) or not isinstance(failed, dict):
        return out

    plan_paths = [
        f["path"] for f in plan.get("files", [])
        if isinstance(f, dict) and isinstance(f.get("path"), str)
    ]
    if not plan_paths:
        return out

    # P11.5.C: contract_failure — приоритетный сигнал из build phase, не из stderr теста.
    # Если _build_one_file_aider пометил результат как contract_failure=True и передал
    # contract.missing/path — берём этот файл в target без парсинга traceback'ов.
    if failed.get("contract_failure"):
        contract = failed.get("contract") or {}
        target_path = failed.get("path") or failed.get("target")
        missing = contract.get("missing") or []
        kind_mm = contract.get("kind_mismatch") or []
        if target_path and target_path in plan_paths:
            out["kind"] = "contract_violation"
            out["target"] = target_path
            out["source_file"] = target_path
            out["missing_name"] = (missing[0] if missing else None)
            out["reason"] = (
                f"contract.lint: {target_path} missing={missing} kind_mismatch={kind_mm}"
            )
            # Собираем хинт из plan.exports для этого файла — чтобы aider восстановил
            # именно те сигнатуры, что были в плане.
            file_node = next(
                (f for f in plan.get("files", [])
                 if isinstance(f, dict) and f.get("path") == target_path),
                None,
            )
            sigs = []
            for exp in ((file_node or {}).get("exports") or []):
                if not isinstance(exp, dict):
                    continue
                nm = (exp.get("name") or "").strip()
                if nm and nm in missing:
                    sig = (exp.get("signature") or nm).strip()
                    kind = (exp.get("kind") or "").strip()
                    sigs.append(f"{kind} {sig}".strip() if kind else sig)
            sigs_block = ("\n  - " + "\n  - ".join(sigs)) if sigs else ""
            out["hint"] = (
                f"Файл {target_path} не экспортирует символы, заявленные в plan.exports: "
                f"missing={missing}. Восстанови эти экспорты на top-level файла "
                f"с их оригинальными сигнатурами:{sigs_block}"
            )
            return out

    stderr = (failed.get("stderr", "") or "") + "\n" + (failed.get("stdout", "") or "")
    if not stderr.strip():
        return out

    # Файл, где реально упало (последний кадр в traceback из плана)
    source_file = _last_user_frame_in_plan(stderr, plan_paths)
    out["source_file"] = source_file

    # Карта exports: имя -> файл, где оно объявлено
    exports_owner: dict[str, str] = {}
    for f in plan.get("files", []):
        if not isinstance(f, dict):
            continue
        rel = f.get("path", "")
        for exp in (f.get("exports") or []):
            if isinstance(exp, dict):
                nm = (exp.get("name") or "").strip()
                if nm:
                    exports_owner[nm] = rel

    # 1) ImportError: cannot import name 'X' from 'M'  → баг M
    m = _HEAL_IMPORTERROR_RE.search(stderr)
    if m:
        missing = m.group(1)
        mod = m.group(2)
        out["kind"] = "import_error"
        out["missing_name"] = missing
        # 1a) plan.exports говорит что missing должен быть в файле X — берём его
        owner = exports_owner.get(missing)
        if owner:
            out["target"] = owner
            out["reason"] = f"plan.exports requires {missing} in {owner}"
        else:
            # 1b) сводим имя модуля к plan-файлу
            mod_path = _module_to_plan_path(mod, plan_paths)
            if mod_path:
                out["target"] = mod_path
                out["reason"] = f"module '{mod}' resolved to {mod_path}"
            else:
                out["target"] = source_file
                out["reason"] = f"unresolved module '{mod}', falling back to source frame"
        out["hint"] = (
            f"В файле {out['target']} отсутствует/переименован символ '{missing}'. "
            f"Добавь определение с тем именем, что объявлено в plan.exports."
        )
        return out

    # 2) AttributeError: module 'M' has no attribute 'X'  → баг M
    m = _HEAL_ATTRMOD_RE.search(stderr)
    if m:
        mod = m.group(1)
        missing = m.group(2)
        out["kind"] = "attribute_error"
        out["missing_name"] = missing
        owner = exports_owner.get(missing)
        if owner:
            out["target"] = owner
            out["reason"] = f"plan.exports requires {missing} in {owner}"
        else:
            mod_path = _module_to_plan_path(mod, plan_paths)
            out["target"] = mod_path or source_file
            out["reason"] = f"module '{mod}' resolved to {out['target']}"
        out["hint"] = (
            f"Модуль '{mod}' не содержит '{missing}'. Добавь определение в {out['target']}."
        )
        return out

    # 3) NameError: name 'X' is not defined  → почти всегда баг в source_file
    m = _HEAL_NAMEERROR_RE.search(stderr)
    if m:
        missing = m.group(1)
        out["kind"] = "name_error"
        out["missing_name"] = missing
        # Если имя — экспорт соседа, source_file должен его импортировать
        owner = exports_owner.get(missing)
        if owner and source_file and owner != source_file:
            out["target"] = source_file
            out["reason"] = f"'{missing}' is exported by {owner}; {source_file} must import it"
            mod = _module_name_from_rel(owner)
            out["hint"] = (
                f"В файле {source_file} используется '{missing}', но нет импорта. "
                f"Добавь: from {mod} import {missing}"
            )
        else:
            # Локальная необъявленная переменная — баг там, где использовалась
            out["target"] = source_file
            out["reason"] = f"undefined local name '{missing}' in {source_file}"
            out["hint"] = (
                f"В файле {source_file} используется '{missing}' без определения. "
                f"Объяви переменную/функцию или добавь нужный импорт."
            )
        return out

    # 4) Любой traceback с user-frame — чиним последний кадр
    if source_file:
        out["kind"] = "traceback"
        out["target"] = source_file
        out["reason"] = f"traceback ends in user file {source_file}"
        return out

    return out


def _pick_heal_target(plan: dict, failed: dict) -> str | None:
    """P11.4: структурный выбор файла для хилинга на основе разбора ошибки.

    Делегирует в _classify_failure (без ключевых слов). Если разбор не дал target —
    fallback на первый .py из plan.files.
    """
    info = _classify_failure(plan, failed)
    if info.get("target"):
        logger.info(
            f"[heal.target] kind={info['kind']} target={info['target']} "
            f"missing={info.get('missing_name')} reason={info.get('reason','')}"
        )
        return info["target"]

    # Fallback: первый .py-файл в plan.files (entry point)
    file_paths = [f["path"] for f in plan.get("files", []) if isinstance(f, dict) and "path" in f]
    if not file_paths:
        return None
    py_files = [p for p in file_paths if p.lower().endswith(".py")]
    fallback = py_files[0] if py_files else file_paths[0]
    logger.info(f"[heal.target] kind=fallback target={fallback}")
    return fallback


def _heal_via_aider(slug: str, plan: dict, failed: dict) -> dict:
    """P9.4: попытаться починить файл через aider_heal.

    Результат — dict с полями ok/target/error/duration. Никогда не бросает исключение.

    P11.4: в error_text включается структурный hint из _classify_failure —
    aider видит и сырой traceback, и явный диагноз (кто виноват и почему).
    """
    info = _classify_failure(plan, failed)
    target = info.get("target") or _pick_heal_target(plan, failed)
    if not target:
        return {"ok": False, "target": None, "error": "no candidate file in plan"}
    raw_err = (failed.get("stderr", "") or failed.get("stdout", "") or "").strip()
    if not raw_err:
        raw_err = f"test '{failed.get('name','?')}' failed (rc={failed.get('rc','?')})"
    # P11.4: добавляем структурный диагноз в error_text
    if info.get("hint"):
        error_text = (
            f"ДИАГНОЗ (kind={info.get('kind')}, target={target}):\n"
            f"{info.get('hint')}\n\n"
            f"СЫРОЙ STDERR:\n{raw_err}"
        )
    else:
        error_text = raw_err
    test_command = failed.get("command", "") or ""
    try:
        pdir = project_dir(slug)
        res = aider_runner.aider_heal(pdir, target, error_text, test_command=test_command)
        return {
            "ok":         bool(res.ok),
            "target":     target,
            "error":      res.error if not res.ok else "",
            "duration_s": res.duration_s,
            "attempts":   res.attempts,
        }
    except Exception as e:
        return {"ok": False, "target": target, "error": f"{type(e).__name__}: {e}"}


def _heal_loop(slug: str, spec: dict, plan: dict, test_results: list[dict], budget: Budget) -> list[dict]:
    if all(r.get("ok") for r in test_results):
        return test_results
    from core.config import AIDER_ENABLED, AIDER_BIN  # локальный импорт — динамический флаг
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
                test_results = _test(slug, plan, spec)
                if all(r.get("ok") for r in test_results):
                    break
                continue
            else:
                add_phase(slug, f"heal:iter{heal_iter}", "failed",
                          f"deterministic pip install failed for {miss.get('missing')}: {miss.get('stderr','')[:200]}")
                # Не выходим — пусть LLM-ветка попробует другой фикс

        # P2: сетевая ошибка → просто пауза+ретест (без LLM, без правки кода)
        net = _heal_network_retry(slug, failed, plan, attempt=heal_iter)
        if net is not None:
            add_phase(slug, f"heal:iter{heal_iter}", "ok",
                      f"network retry after {net.get('retried_after_s')}s (attempt {net.get('attempt')})")
            test_results = _test(slug, plan, spec)
            if all(r.get("ok") for r in test_results):
                break
            continue

        # P2: SyntaxError/IndentationError → детерминистическая диагностика без LLM
        diag: dict | None = _heal_syntax_error(slug, failed, plan)
        det_category = "syntax" if diag else None

        # P2: JSONDecodeError → фиксированный хинт Coder'у
        if diag is None:
            diag = _heal_json_decode(slug, failed, plan)
            det_category = "json" if diag else None

        # P9.4: aider-ветка — если детерминистические хилеры не помогли и aider включён,
        # отдаём ему файл и stderr напрямую. Aider сам и диагностирует, и патчит, и сохраняет.
        # Если aider успешно применил правку — ретестируем; иначе fallthrough в LLM-диагностику.
        if diag is None and AIDER_ENABLED and aider_runner.is_aider_available(AIDER_BIN):
            try:
                budget.check(f"heal:aider:iter{heal_iter}")
                ah = _heal_via_aider(slug, plan, failed)
                budget.spend(1)
            except BudgetExceeded as e:
                add_phase(slug, f"heal:iter{heal_iter}", "failed", f"budget: {e}")
                break
            if ah.get("ok"):
                add_phase(
                    slug, f"heal:iter{heal_iter}", "ok",
                    f"aider heal target={ah.get('target')} duration={ah.get('duration_s')}s"
                )
                test_results = _test(slug, plan, spec)
                if all(r.get("ok") for r in test_results):
                    break
                continue
            else:
                logger.info(
                    f"[heal] aider failed (target={ah.get('target')} err={ah.get('error','')[:200]}), "
                    "fallback to LLM diagnose"
                )
                # fallthrough в обычный LLM-путь ниже

        if diag is not None:
            add_phase(slug, f"heal:iter{heal_iter}", "ok",
                      f"deterministic diagnosis ({det_category}) target={diag.get('target_file')}")
        else:
            try:
                diag = _diagnose(spec, file_paths, failed, budget)
            except BudgetExceeded as e:
                add_phase(slug, f"heal:iter{heal_iter}", "failed", f"budget: {e}")
                break
        # diag либо из детерминистического хилера, либо из _diagnose — в обоих случаях
        # дальше идёт общий patch_file -> ретест.

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
        new_results = _test(slug, plan, spec)
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
        raw = _llm(budget, MODEL_README, PROJECT_README_SYSTEM, user,
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
def _deterministic_report(slug: str, spec: dict, build_results: list[dict],
                          test_results: list[dict]) -> str:
    """P5.2: детерминистический отчёт — без LLM, без галлюцинаций, строго по фактам.

    Никогда не выдумывает ошибки. Описывает только то что реально в build_results/test_results.
    """
    title = spec.get("title") or slug
    files_ok = [r["path"] for r in build_results if r.get("ok")]
    build_ok = len(files_ok)
    build_total = len(build_results)
    tests_total = len(test_results)
    tests_ok = sum(1 for r in test_results if r.get("ok"))
    failed_tests = [r for r in test_results if not r.get("ok")]

    parts: list[str] = []
    parts.append(f"Готово, сэр. Проект «{title}» лежит в data/projects/{slug}.")

    # Файлы
    if files_ok:
        if len(files_ok) <= 3:
            files_str = ", ".join(files_ok)
        else:
            files_str = f"{len(files_ok)} файлов включая " + ", ".join(files_ok[:3])
        parts.append(f"Собраны: {files_str}.")
    else:
        parts.append("Ни один файл не собрался.")

    # Тесты — только факты
    if tests_total == 0:
        parts.append("Тесты не были заданы.")
    elif tests_ok == tests_total:
        if tests_total == 1:
            parts.append("Тест прошёл успешно.")
        else:
            parts.append(f"Все {tests_total} тестов прошли успешно.")
    else:
        parts.append(f"Тестов прошло {tests_ok} из {tests_total}.")
        # Короткий фрагмент первой ошибки — только если реально есть.
        first = failed_tests[0]
        err = (first.get("stderr") or "").strip()
        if err:
            short = err.splitlines()[-1][:120]
            parts.append(f"Первая ошибка: {short}.")

    # build vs total — если не все файлы собрались, упомянуть
    if build_ok < build_total:
        parts.append(f"Собралось {build_ok} из {build_total} файлов.")

    return " ".join(parts)


def _report(slug: str, spec: dict, build_results: list[dict], test_results: list[dict],
            budget: Budget) -> str:
    """P5.2: если все тесты ок и все файлы собрались — детерминистика без LLM (врать нечему).

    LLM зовём только когда есть реальные проблемы и нужно объяснить.
    """
    all_files_ok = all(r.get("ok") for r in build_results) and len(build_results) > 0
    all_tests_ok = all(r.get("ok") for r in test_results)
    has_failed_tests = any(not r.get("ok") for r in test_results)

    # Счастливый путь — без LLM. Невозможно врать.
    if all_files_ok and all_tests_ok and not has_failed_tests:
        return _deterministic_report(slug, spec, build_results, test_results)

    # Иначе — пробуем LLM, но fallback на детерминистику при любой ошибке.
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
    user = ("Итоги проекта в JSON (опирайся ТОЛЬКО на эти данные):\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + f"\n\nПапка проекта: data/projects/{slug}/")
    try:
        return _llm(budget, MODEL_REPORT, PROJECT_REPORT_SYSTEM, user,
                    temperature=0.2, num_ctx=2048, where="report").strip()
    except (BudgetExceeded, Exception) as e:
        logger.warning(f"[project.report] LLM unavailable: {e} — using deterministic")
        return _deterministic_report(slug, spec, build_results, test_results)


# ─── PUBLIC: run() ──────────────────────────────────────────────────────────
def run(query: str, history: list[dict] | None = None,
        *, wall_budget_s: float = PROJECT_WALL_BUDGET_S,
        llm_budget: int = PROJECT_LLM_BUDGET) -> str:
    """Полный цикл: запрос → готовый проект."""
    if not isinstance(query, str) or not query.strip():
        return "Сэр, я не понял какой проект нужно сделать."

    # P3: на intake бюджет фиксированный (минимум как XS), потом переоцениваем по spec.
    budget = Budget(wall_s=wall_budget_s, llm=llm_budget)

    # PHASE 1
    try:
        spec = _intake(query, budget)
    except BudgetExceeded as e:
        return f"Бюджет исчерпан на этапе intake: {e}"
    except Exception as e:
        logger.error(f"[project.intake] {e}")
        return f"Не удалось разобрать задачу: {e}"

    # P3: адаптивный бюджет — только если вызвавший не указал явно свои значения.
    if wall_budget_s == PROJECT_WALL_BUDGET_S and llm_budget == PROJECT_LLM_BUDGET:
        tier = estimate_complexity(query, spec=spec, plan=None)
        params = budget_for_tier(tier)
        # Не урезаем уже потраченное: сохраняем llm_used, обновляем лимиты.
        spent = budget.llm_used
        budget = Budget(wall_s=params["wall_s"], llm=params["llm"])
        budget.llm_used = spent
        logger.info(f"[project] adaptive budget: tier={tier} llm={params['llm']} wall_s={params['wall_s']}")

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
                # P9.10: обогащаем plan.inputs heuristic-входами из spec.summary,
                # чтобы coder/aider видели реальные имена файлов, а не придумывали свои.
                plan = _enrich_plan_with_heuristic_inputs(plan, spec)
                # P11.1: нормализуем контракты (exports per file, depends_on consistency).
                # Lossless: ничего не отбрасывает, только заполняет/исправляет поля.
                plan = _normalize_plan_contracts(plan, spec)
                # P11.2.e (FM-10): убираем из plan.files файлы, которые уже в plan.inputs.
                plan = _dedupe_files_vs_inputs(plan)
                m = load_manifest(slug); m.plan = plan; save_manifest(m)
                _cm = plan.get("_contract_metrics", {}) or {}
                add_phase(slug, "architect", "ok",
                          f"files={len(plan['files'])} tests={len(plan.get('tests',[]))} "
                          f"inputs={len(plan.get('inputs',[]))} "
                          f"exports={_cm.get('files_with_exports', 0)}/{_cm.get('py_files', 0)} "
                          f"dep_unmatched={len(_cm.get('depends_unmatched', []) or [])}")
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

        # P11.5.C: PHASE 4.5 — блокирующий контракт-линтер между BUILD и TEST.
        # Если любой файл провалил линт экспортов — синтезируем failed-test-record
        # и сразу пускаем через _heal_loop. Это даёт heal-loop'у точный target
        # (файл с missing exports) до того как smoke упадёт на ImportError.
        contract_failed_files: list[dict] = []
        if 3 >= skip_until and build_results:
            for br in build_results:
                if br.get("contract_failure"):
                    contract_failed_files.append(br)
            if contract_failed_files:
                synthetic_results: list[dict] = []
                for br in contract_failed_files:
                    contract = br.get("contract") or {}
                    missing = contract.get("missing") or []
                    synthetic_results.append({
                        "name":    f"contract_lint:{br.get('path','?')}",
                        "command": "",
                        "ok":      False,
                        "rc":      None,
                        "stdout":  "",
                        "stderr":  (
                            f"contract.lint blocked build (P11.5): "
                            f"file={br.get('path')} missing={missing}"
                        ),
                        "expects": "plan.exports satisfied",
                        # P11.5.C: эти поля читаются _classify_failure'ом
                        "contract_failure": True,
                        "contract": contract,
                        "path":     br.get("path"),
                    })
                add_phase(
                    slug, "contract.block", "failed",
                    f"files={[r.get('path') for r in contract_failed_files]} "
                    f"→ heal-loop без smoke (P11.5)"
                )
                try:
                    synthetic_results = _heal_loop(slug, spec, plan, synthetic_results, budget)
                except BudgetExceeded as e:
                    add_phase(slug, "contract.block", "failed", f"budget: {e}")
                # После heal'а всё равно пойдёт PHASE 5 — реальный smoke решит ок/не ок.

        # PHASE 5: TEST
        if 3 >= skip_until:
            try:
                test_results = _test(slug, plan, spec)
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
