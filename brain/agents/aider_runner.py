"""P9: aider как builder/healer для ProjectAgent.

aider — зрелый CLI coding-агент, который умеет редактировать файлы по инструкции
с учётом синтаксиса (search/replace blocks, retry на parse error). Работает с
локальным ollama через openai-совместимый endpoint.

Контракт: одна функция `aider_build(...)` и одна `aider_heal(...)`. Обе возвращают
BuildResult — единый dict, который ProjectAgent уже умеет обрабатывать в _build_one_file.

fix #5: is_aider_available() теперь:
  - Возвращает False без subprocess если AIDER_ENABLED=False
  - Кеширует результат проверки чтобы не запускать subprocess при каждом проекте

fix BUG-1: кэш теперь dict[str, bool] с ключом bin_path, а не глобальный bool.
  - Хранение по bin_path устраняет баг: если is_aider_available('aider') вернула False,
    а потом is_aider_available('/path/to/aider') не возвращала устаревший кэш.
  - invalidate_aider_cache(bin_path=None) добавлен для принудительной перепроверки
    (например, после установки aider в время работы сессии).
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

# fix BUG-1: кэш теперь dict[bin_path, bool], а не один bool.
# При первом вызове с bin_path='aider' и потом с '/opt/venv/bin/aider'
# каждый проверяется и кэшируется независимо.
_aider_available_cache: dict[str, bool] = {}


@dataclass
class AiderResult:
    """Результат вызова aider. Единая структура для build и heal."""
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
    """Собрать аргументы для aider CLI."""
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
    """Быстрая проверка: есть ли aider в PATH.

    fix #5:
      - Если AIDER_ENABLED=False — сразу False без subprocess (экономия ~10с на проект).
      - Кеширует результат проверки на время жизни процесса — subprocess вызывается всего один раз.

    fix BUG-1:
      - Кэш теперь dict[bin_path → bool], а не один глобальный bool.
      - При разных bin_path (имя + PATH vs абсолютный путь) каждый проверяется верно.
      - invalidate_aider_cache() позволяет перепроверить после доустановки aider.
    """
    # Быстрый путь: если выключен в config — не тратим время на subprocess
    if not AIDER_ENABLED:
        return False

    # fix BUG-1: кэш по bin_path
    if bin_path in _aider_available_cache:
        return _aider_available_cache[bin_path]

    try:
        result = subprocess.run(
            [bin_path, "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        _aider_available_cache[bin_path] = (result.returncode == 0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _aider_available_cache[bin_path] = False

    return _aider_available_cache[bin_path]


def invalidate_aider_cache(bin_path: str | None = None) -> None:
    """Принудительно сбросить кэш is_aider_available().

    bin_path=None — сбросить весь кэш (все bin_path).
    bin_path=str — сбросить кэш только для этого пути.

    Пример использования: после `pip install aider-chat` в рабочей сессии:
        from brain.agents.aider_runner import invalidate_aider_cache
        invalidate_aider_cache()
    """
    if bin_path is None:
        _aider_available_cache.clear()
    else:
        _aider_available_cache.pop(bin_path, None)
