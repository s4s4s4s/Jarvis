"""P9: aider как builder/healer для ProjectAgent.

aider — зрелый CLI coding-агент, который умеет редактировать файлы по инструкции
с учётом синтаксиса (search/replace blocks, retry на parse error). Работает с
локальным ollama через openai-совместимый endpoint.

Контракт: одна функция `aider_build(...)` и одна `aider_heal(...)`. Обе возвращают
BuildResult — единый dict, который ProjectAgent уже умеет обрабатывать в _build_one_file.

fix #5: is_aider_available() теперь:
  - Возвращает False без subprocess если AIDER_ENABLED=False
  - Кеширует результат проверки чтобы не запускать subprocess при каждом проекте
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import (
    AIDER_API_BASE,
    AIDER_BIN,
    AIDER_ENABLED,
    AIDER_MAX_RETRIES,
    AIDER_MODEL,
    AIDER_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

# fix #5: кэш результата is_aider_available(). None = ещё не проверяли.
_aider_available_cache: bool | None = None


@dataclass
class AiderResult:
    """\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0432\u044b\u0437\u043e\u0432\u0430 aider. \u0415\u0434\u0438\u043d\u0430\u044f \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0430 \u0434\u043b\u044f build \u0438 heal."""
    ok: bool
    file_path: str
    content: str
    duration_s: float
    error: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    attempts: int = 1


def _build_argv(
    project_dir: Path,
    target_file: str,
    instruction: str,
    *,
    model: str,
    api_base: str,
    extra_files: list[str] | None = None,
    read_only_files: list[str] | None = None,
) -> list[str]:
    """\u0421\u043e\u0431\u0440\u0430\u0442\u044c \u0430\u0440\u0433\u0443\u043c\u0435\u043d\u0442\u044b \u0434\u043b\u044f aider CLI."""
    argv = [
        AIDER_BIN,
        "--model", model,
        "--no-git",
        "--no-auto-commits",
        "--yes-always",
        "--no-pretty",
        "--no-stream",
        "--no-check-update",
        "--no-show-model-warnings",
        "--message", instruction,
        target_file,
    ]
    for f in (extra_files or []):
        argv.append(f)
    for f in (read_only_files or []):
        argv.extend(["--read", f])
    return argv


def _aider_env(api_base: str) -> dict:
    env = os.environ.copy()
    base = api_base.rstrip("/")
    env["OLLAMA_API_BASE"] = base[:-3] if base.endswith("/v1") else base
    env["OPENAI_API_BASE"] = base if base.endswith("/v1") else base + "/v1"
    env.setdefault("OPENAI_API_KEY", "ollama-local")
    env.setdefault("AIDER_ANALYTICS", "false")
    return env


def _run_subprocess(
    argv: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict,
) -> tuple[int, str, str, float]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv, cwd=str(cwd), env=env,
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        return (completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
                time.monotonic() - started)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return (-1, out, err + f"\n[TIMEOUT after {timeout_s}s]", time.monotonic() - started)
    except FileNotFoundError as e:
        return (-2, "", f"aider binary not found: {e}", time.monotonic() - started)


def _read_file_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[aider_runner] failed to read {path}: {e}")
        return ""


def aider_build(
    project_dir: Path,
    target_file: str,
    instruction: str,
    *,
    model: Optional[str] = None,
    timeout_s: Optional[int] = None,
    max_retries: Optional[int] = None,
    api_base: Optional[str] = None,
    extra_files: list[str] | None = None,
    read_only_files: list[str] | None = None,
) -> AiderResult:
    model = model or AIDER_MODEL
    timeout_s = timeout_s or AIDER_TIMEOUT_S
    max_retries = max_retries if max_retries is not None else AIDER_MAX_RETRIES
    api_base = api_base or AIDER_API_BASE

    target_path = project_dir / target_file
    project_dir.mkdir(parents=True, exist_ok=True)

    argv = _build_argv(project_dir, target_file, instruction,
                       model=model, api_base=api_base, extra_files=extra_files,
                       read_only_files=read_only_files)
    env = _aider_env(api_base)

    last_stdout, last_stderr = "", ""
    last_rc = 0
    total_duration = 0.0
    attempts = 0

    for attempt in range(1, max_retries + 2):
        attempts = attempt
        logger.info(f"[aider.build] attempt {attempt}/{max_retries + 1} target={target_file} model={model}")
        rc, stdout, stderr, duration = _run_subprocess(argv, project_dir, timeout_s, env)
        total_duration += duration
        last_rc, last_stdout, last_stderr = rc, stdout, stderr

        content = _read_file_safely(target_path)
        if rc == 0 or (rc != -2 and content.strip()):
            return AiderResult(
                ok=bool(content.strip()),
                file_path=target_file,
                content=content,
                duration_s=round(total_duration, 2),
                error="" if content.strip() else "aider exited cleanly but file is empty",
                stdout=stdout[-4000:],
                stderr=stderr[-2000:],
                exit_code=rc,
                attempts=attempts,
            )

        if rc == -2:
            break
        logger.warning(
            f"[aider.build] attempt {attempt} rc={rc}; retrying"
            if attempt <= max_retries else
            f"[aider.build] attempt {attempt} rc={rc}; giving up"
        )

    return AiderResult(
        ok=False,
        file_path=target_file,
        content=_read_file_safely(target_path),
        duration_s=round(total_duration, 2),
        error=f"aider failed after {attempts} attempts (last rc={last_rc})",
        stdout=last_stdout[-4000:],
        stderr=last_stderr[-2000:],
        exit_code=last_rc,
        attempts=attempts,
    )


def aider_heal(
    project_dir: Path,
    target_file: str,
    error_text: str,
    *,
    test_command: str = "",
    model: Optional[str] = None,
    timeout_s: Optional[int] = None,
    max_retries: Optional[int] = None,
    api_base: Optional[str] = None,
    read_only_files: list[str] | None = None,
) -> AiderResult:
    instruction_parts = [
        f"При проверке файла {target_file} возникла ошибка."
    ]
    if test_command:
        instruction_parts.append(f"Команда запуска: {test_command}")
    instruction_parts.append(f"Текст ошибки:\n{error_text.strip()[:2000]}")
    instruction_parts.append(
        "Исправь файл так чтобы ошибка ушла. "
        "Не добавляй комментарии-извинения, не меняй структуру если этого не требует ошибка."
    )
    instruction = "\n\n".join(instruction_parts)

    return aider_build(
        project_dir, target_file, instruction,
        model=model, timeout_s=timeout_s, max_retries=max_retries,
        api_base=api_base, read_only_files=read_only_files,
    )


def is_aider_available(bin_path: str = AIDER_BIN) -> bool:
    """\u0411\u044b\u0441\u0442\u0440\u0430\u044f \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430: \u0435\u0441\u0442\u044c \u043b\u0438 aider \u0432 PATH.

    fix #5:
      - Е\u0441\u043b\u0438 AIDER_ENABLED=False \u2014 \u0441\u0440\u0430\u0437\u0443 False \u0431\u0435\u0437 subprocess (\u044d\u043a\u043e\u043d\u043e\u043c\u0438\u044f ~10\u0441 \u043d\u0430 \u043f\u0440\u043e\u0435\u043a\u0442).
      - \u041a\u0435\u0448\u0438\u0440\u0443\u0435\u0442 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u043d\u0430 \u0432\u0440\u0435\u043c\u044f \u0436\u0438\u0437\u043d\u0438 \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u0430 \u2014 subprocess \u0432\u044b\u0437\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u0432\u0441\u0435\u0433\u043e \u043e\u0434\u0438\u043d \u0440\u0430\u0437.
    """
    global _aider_available_cache

    # Б\u044b\u0441\u0442\u0440\u044b\u0439 \u043f\u0443\u0442\u044c: \u0435\u0441\u043b\u0438 \u0432\u044b\u043a\u043b\u044e\u0447\u0435\u043d \u0432 config \u2014 \u043d\u0435 \u0442\u0440\u0430\u0442\u0438\u043c \u0432\u0440\u0435\u043c\u044f \u043d\u0430 subprocess
    if not AIDER_ENABLED:
        return False

    # \u041a\u0435\u0448 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430 \u0434\u043b\u044f \u0442\u0435\u043a\u0443\u0449\u0435\u0433\u043e \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u0430
    if _aider_available_cache is not None:
        return _aider_available_cache

    try:
        result = subprocess.run(
            [bin_path, "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        _aider_available_cache = (result.returncode == 0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _aider_available_cache = False

    return _aider_available_cache
