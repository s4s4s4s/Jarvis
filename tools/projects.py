"""
tools/projects.py — стор и операции над проектами Level 4.

Каждый проект — папка PROJECTS_DIR/<slug>/ с:
  - manifest.json     — метаданные, спецификация, история фаз
  - <код проекта>     — файлы которые сгенерировал ProjectAgent
  - logs/             — логи запусков тестов и smoke-runs

Никакой LLM здесь нет. Это чисто инфраструктура хранения и безопасных операций
с файловой системой В ПРЕДЕЛАХ ПАПКИ ПРОЕКТА.

Используется ProjectAgent (brain/agents/project.py).
Может быть зарегистрирован как tool позже (project.list / project.status), сейчас — нет.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import PROJECTS_DIR, PROJECT_LOG

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
MAX_SLUG_LEN  = 40
SAFE_FILE_RE  = re.compile(r"^[A-Za-z0-9_./-]+$")
MAX_FILE_BYTES = 200_000          # 200 KB на один файл
MAX_TOTAL_BYTES = 5_000_000       # 5 MB на проект целиком
SUBPROCESS_TIMEOUT = 30           # секунд на один запуск
VENV_DIR_NAME = ".venv"
PIP_INSTALL_TIMEOUT = 180         # 3 мин на pip install
ALLOWED_PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}([<>=!~]=?[A-Za-z0-9._-]+)?$")


# ─── slug ────────────────────────────────────────────────────────────────────
def slugify(title: str) -> str:
    """Normalize an LLM-suggested slug to safe charset."""
    s = title.strip().lower()
    # транслитерация минимально-разумная — кириллица → латиница
    table = str.maketrans({
        "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e",
        "ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m",
        "н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u",
        "ф":"f","х":"h","ц":"c","ч":"ch","ш":"sh","щ":"sch",
        "ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya",
        " ":"-","_":"-",
    })
    s = s.translate(table)
    s = _SLUG_RE.sub("-", s).strip("-")
    s = re.sub(r"-+", "-", s)
    if not s:
        s = f"project-{int(time.time())}"
    return s[:MAX_SLUG_LEN]


# ─── манифест ─────────────────────────────────────────────────────────────────
@dataclass
class PhaseRecord:
    name: str
    status: str                   # "ok" | "failed" | "skipped"
    detail: str = ""
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ProjectManifest:
    slug: str
    title: str
    kind: str = "script"
    language: str = "python"
    summary: str = ""
    spec: dict = field(default_factory=dict)        # полная intake-спецификация
    plan: dict = field(default_factory=dict)        # архитектурный план
    files: list[str] = field(default_factory=list)  # файлы которые мы создали
    phases: list[dict] = field(default_factory=list)
    status: str = "in_progress"                     # "in_progress"|"done"|"failed"
    created_at: str = ""
    updated_at: str = ""
    request: str = ""                               # исходный запрос пользователя
    metrics: dict = field(default_factory=dict)     # llm_calls, durations, heal_iters
    last_phase: str = ""                            # последняя успешно пройденная фаза (для resume)

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.created_at:
            self.created_at = now
        self.updated_at = now


# ─── helpers ─────────────────────────────────────────────────────────────────
def _project_dir(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{1,%d}" % MAX_SLUG_LEN, slug):
        raise ValueError(f"unsafe slug: {slug!r}")
    return PROJECTS_DIR / slug


def _manifest_path(slug: str) -> Path:
    return _project_dir(slug) / "manifest.json"


def _safe_resolve(slug: str, rel: str) -> Path:
    """Resolve a project-relative path and forbid escape via .. or absolute."""
    if not rel or not isinstance(rel, str):
        raise ValueError("empty path")
    norm = rel.replace("\\", "/")
    # absolute paths are forbidden BEFORE any stripping
    if norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        raise ValueError(f"absolute path forbidden: {rel!r}")
    if not SAFE_FILE_RE.match(norm):
        raise ValueError(f"unsafe path: {rel!r}")
    if ".." in norm.split("/"):
        raise ValueError(f"path traversal: {rel!r}")
    base = _project_dir(slug).resolve()
    full = (base / norm).resolve()
    if base not in full.parents and full != base:
        raise ValueError(f"path escapes project root: {rel!r}")
    return full


# ─── public API ─────────────────────────────────────────────────────────────
def create_project(spec: dict) -> ProjectManifest:
    """
    Create new project directory and persist manifest.
    spec is the JSON dict from PROJECT_INTAKE_SYSTEM.
    """
    title = (spec.get("title") or "").strip() or "untitled-project"
    raw_slug = (spec.get("slug") or "").strip().lower()
    slug = slugify(raw_slug or title)

    # collision: append timestamp suffix
    if (PROJECTS_DIR / slug).exists():
        slug = f"{slug}-{int(time.time())}"[:MAX_SLUG_LEN]

    pdir = PROJECTS_DIR / slug
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "logs").mkdir(exist_ok=True)

    manifest = ProjectManifest(
        slug=slug,
        title=title,
        kind=str(spec.get("kind") or "script"),
        language=str(spec.get("language") or "python"),
        summary=str(spec.get("summary") or ""),
        spec=spec,
    )
    save_manifest(manifest)
    _journal({"event": "create", "slug": slug, "title": title})
    return manifest


def save_manifest(m: ProjectManifest) -> None:
    m.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = _manifest_path(m.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(m), ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(slug: str) -> ProjectManifest:
    data = json.loads(_manifest_path(slug).read_text(encoding="utf-8"))
    return ProjectManifest(**data)


def list_projects() -> list[dict]:
    if not PROJECTS_DIR.exists():
        return []
    out = []
    for p in sorted(PROJECTS_DIR.iterdir()):
        mp = p / "manifest.json"
        if not mp.exists():
            continue
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            out.append({
                "slug": data.get("slug"),
                "title": data.get("title"),
                "status": data.get("status"),
                "updated_at": data.get("updated_at"),
            })
        except Exception as e:
            logger.warning(f"[projects] bad manifest {mp}: {e}")
    return out


def write_project_file(slug: str, rel_path: str, content: str) -> dict:
    """Write a file inside the project. Returns {ok, path, bytes}."""
    if not isinstance(content, str):
        raise TypeError("content must be str")
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError(f"file too large (> {MAX_FILE_BYTES} bytes)")
    full = _safe_resolve(slug, rel_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")

    # обновим manifest.files
    try:
        m = load_manifest(slug)
        if rel_path not in m.files:
            m.files.append(rel_path)
            save_manifest(m)
    except Exception as e:
        logger.warning(f"[projects] could not update files list: {e}")

    return {"ok": True, "path": str(full), "bytes": full.stat().st_size}


def read_project_file(slug: str, rel_path: str) -> str:
    full = _safe_resolve(slug, rel_path)
    if not full.exists():
        raise FileNotFoundError(rel_path)
    return full.read_text(encoding="utf-8")


def add_phase(slug: str, name: str, status: str, detail: str = "") -> None:
    m = load_manifest(slug)
    m.phases.append(asdict(PhaseRecord(name=name, status=status, detail=detail)))
    save_manifest(m)
    _journal({"event": "phase", "slug": slug, "name": name, "status": status, "detail": detail[:300]})


def set_status(slug: str, status: str) -> None:
    m = load_manifest(slug)
    m.status = status
    save_manifest(m)


def run_in_project(slug: str, cmd: list[str], timeout: int = SUBPROCESS_TIMEOUT) -> dict:
    """
    Run a shell command WITH cwd inside project dir, NEVER shell=True.
    Returns {ok, returncode, stdout, stderr, timed_out}.
    """
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        raise TypeError("cmd must be list[str]")
    if not cmd:
        raise ValueError("empty cmd")
    pdir = _project_dir(slug)
    log_path = pdir / "logs" / f"run-{int(time.time())}.log"
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(pdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        rc  = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        rc  = -1
    except FileNotFoundError as e:
        out, err, rc = "", f"executable not found: {e}", -1

    log_path.write_text(
        f"$ {' '.join(cmd)}\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n--- rc={rc} timed_out={timed_out} ---\n",
        encoding="utf-8",
    )
    return {
        "ok": (rc == 0 and not timed_out),
        "returncode": rc,
        "stdout": out[-4000:],
        "stderr": err[-4000:],
        "timed_out": timed_out,
        "log": str(log_path),
    }


def python_smoke(slug: str, entry: str, timeout: int = SUBPROCESS_TIMEOUT) -> dict:
    """Run `python <entry>` inside project. Convenience wrapper."""
    return run_in_project(slug, [sys.executable, entry], timeout=timeout)


def _journal(rec: dict) -> None:
    rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        PROJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PROJECT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[projects] journal write failed: {e}")


# ─── venv и зависимости ────────────────────────────────────────────────────
def _venv_python(slug: str) -> Path:
    """Путь к python внутри изолированного venv. Работает на Windows и POSIX."""
    pdir = _project_dir(slug)
    if sys.platform == "win32":
        return pdir / VENV_DIR_NAME / "Scripts" / "python.exe"
    return pdir / VENV_DIR_NAME / "bin" / "python"


def ensure_venv(slug: str) -> dict:
    """Создать venv в папке проекта если его нет. Идемпотентно."""
    py = _venv_python(slug)
    if py.exists():
        return {"ok": True, "created": False, "python": str(py)}
    pdir = _project_dir(slug)
    try:
        # python -m venv .venv  — без pip-upgrade чтобы было быстро и оффлайн-дружелюбно
        proc = subprocess.run(
            [sys.executable, "-m", "venv", VENV_DIR_NAME],
            cwd=str(pdir),
            capture_output=True, text=True, timeout=60, shell=False, check=False,
        )
        if proc.returncode != 0 or not py.exists():
            return {"ok": False, "error": (proc.stderr or proc.stdout or "venv create failed")[-500:]}
        return {"ok": True, "created": True, "python": str(py)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "venv creation timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _validate_pkg_spec(pkg: str) -> bool:
    return bool(ALLOWED_PKG_RE.match(pkg or ""))


def pip_install(slug: str, packages: list[str], timeout: int = PIP_INSTALL_TIMEOUT) -> dict:
    """Установить пакеты в venv проекта. Никаких -e/--user/git+ — только PyPI-имена."""
    if not isinstance(packages, list) or not packages:
        return {"ok": True, "installed": [], "skipped": True}
    bad = [p for p in packages if not isinstance(p, str) or not _validate_pkg_spec(p)]
    if bad:
        return {"ok": False, "error": f"unsafe package spec: {bad}"}
    ev = ensure_venv(slug)
    if not ev["ok"]:
        return {"ok": False, "error": f"venv: {ev.get('error')}"}
    py = _venv_python(slug)
    cmd = [str(py), "-m", "pip", "install", "--no-input", "--disable-pip-version-check", *packages]
    try:
        proc = subprocess.run(
            cmd, cwd=str(_project_dir(slug)),
            capture_output=True, text=True, timeout=timeout, shell=False, check=False,
        )
        ok = (proc.returncode == 0)
        log_path = _project_dir(slug) / "logs" / f"pip-{int(time.time())}.log"
        log_path.write_text(
            f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n--- rc={proc.returncode} ---\n",
            encoding="utf-8",
        )
        return {
            "ok": ok,
            "installed": packages if ok else [],
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-1500:],
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "pip install timed out", "installed": []}
    except Exception as e:
        return {"ok": False, "error": str(e), "installed": []}


def run_with_project_python(slug: str, args: list[str], timeout: int = SUBPROCESS_TIMEOUT) -> dict:
    """Запуск команды через venv-python (или sys.executable если venv не создан)."""
    py = _venv_python(slug)
    interp = str(py) if py.exists() else sys.executable
    return run_in_project(slug, [interp, *args], timeout=timeout)


def run_pytest(slug: str, timeout: int = SUBPROCESS_TIMEOUT) -> dict:
    """Запустить pytest внутри проекта (если pytest установлен в venv)."""
    return run_with_project_python(slug, ["-m", "pytest", "-q", "--no-header", "--tb=short"], timeout=timeout)


# ─── индекс и метрики ────────────────────────────────────────────────────────
def _index_path() -> Path:
    return PROJECTS_DIR / "_index.jsonl"


def append_index_record(rec: dict) -> None:
    """Дописать одну строку в _index.jsonl. Не падаем если файловая ошибка."""
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        with _index_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[projects] _index write failed: {e}")


def read_index() -> list[dict]:
    p = _index_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def get_project_files(slug: str) -> dict[str, str]:
    """Прочитать все сохранённые файлы проекта в dict {rel_path: content}. Исключает venv и logs."""
    out: dict[str, str] = {}
    try:
        m = load_manifest(slug)
    except Exception:
        return out
    for rel in m.files:
        try:
            out[rel] = read_project_file(slug, rel)
        except Exception:
            continue
    return out
