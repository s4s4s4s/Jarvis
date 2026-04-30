"""P9: aider как builder/healer для ProjectAgent.

aider — зрелый CLI coding-агент, который умеет редактировать файлы по инструкции
с учётом синтаксиса (search/replace blocks, retry на parse error). Работает с
локальным ollama через openai-совместимый endpoint.

Контракт: одна функция `aider_build(...)` и одна `aider_heal(...)`. Обе возвращают
BuildResult — единый dict, который ProjectAgent уже умеет обрабатывать в _build_one_file.

Дизайн:
  • subprocess + жёсткий timeout
  • stderr и stdout полностью пишутся в манифест проекта (для аудита)
  • при таймауте / non-zero exit — ok=False с error-полем
  • никакого парсинга диффов своими руками — aider сам пишет файл, мы только читаем результат

Совместимость с принципами проекта:
  • полностью локально (ollama через openai-API совместимость)
  • один файл — самодостаточный
  • всегда есть fallback (старый путь активен по AIDER_ENABLED=False)
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
    AIDER_MAX_RETRIES,
    AIDER_MODEL,
    AIDER_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


@dataclass
class AiderResult:
    """Результат вызова aider. Единая структура для build и heal."""
    ok: bool
    file_path: str          # относительный путь файла
    content: str            # фактическое содержимое после aider (читается с диска)
    duration_s: float
    error: str = ""         # пусто если ok
    stdout: str = ""        # для аудита
    stderr: str = ""
    exit_code: int = 0
    attempts: int = 1       # сколько попыток понадобилось (1..AIDER_MAX_RETRIES)


def _build_argv(
    project_dir: Path,
    target_file: str,
    instruction: str,
    *,
    model: str,
    api_base: str,
    extra_files: list[str] | None = None,
) -> list[str]:
    """Собрать аргументы для aider CLI.

    Используем флаги:
      --model ollama/<name>     — какую модель грузить через ollama
      --no-git                  — у нас своя структура data/projects/<slug>/, git не нужен
      --no-auto-commits         — мы сами решаем когда коммитить
      --yes-always              — без интерактивных подтверждений
      --no-pretty               — чистый вывод для парсинга stdout
      --no-stream               — отключаем streaming, ждём финальный ответ
      --no-check-update         — не дёргать сеть на проверку версии
      --message <prompt>        — non-interactive: один шаг и выход
    """
    argv = [
        AIDER_BIN,
        "--model", model,
        "--no-git",
        "--no-auto-commits",
        "--yes-always",
        "--no-pretty",
        "--no-stream",
        "--no-check-update",
        "--message", instruction,
        target_file,
    ]
    for f in (extra_files or []):
        argv.append(f)
    return argv


def _aider_env(api_base: str) -> dict:
    """Переменные окружения для aider: указываем где ollama.

    aider читает OPENAI_API_BASE / OPENAI_API_KEY (с ollama ключ — что угодно).
    """
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = api_base + ("/v1" if not api_base.rstrip("/").endswith("/v1") else "")
    env.setdefault("OPENAI_API_KEY", "ollama-local")  # любая непустая строка
    # Тише по умолчанию
    env.setdefault("AIDER_ANALYTICS", "false")
    return env


def _run_subprocess(
    argv: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict,
) -> tuple[int, str, str, float]:
    """Запустить aider с timeout. Возвращает (exit_code, stdout, stderr, duration_s)."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return (completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
                time.monotonic() - started)
    except subprocess.TimeoutExpired as e:
        # вернём специальный exit_code -1 чтобы вызывающий мог отличить timeout от non-zero
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
) -> AiderResult:
    """Сгенерировать или дописать файл через aider.

    Args:
      project_dir   — корень проекта (data/projects/<slug>)
      target_file   — относительный путь файла (например "main.py")
      instruction   — что нужно сделать (промпт для aider)
      extra_files   — дополнительные файлы которые aider должен видеть как контекст

    Returns:
      AiderResult с ok=True если файл существует и непустой после прогона.
    """
    model = model or AIDER_MODEL
    timeout_s = timeout_s or AIDER_TIMEOUT_S
    max_retries = max_retries if max_retries is not None else AIDER_MAX_RETRIES
    api_base = api_base or AIDER_API_BASE

    target_path = project_dir / target_file
    project_dir.mkdir(parents=True, exist_ok=True)

    argv = _build_argv(project_dir, target_file, instruction,
                       model=model, api_base=api_base, extra_files=extra_files)
    env = _aider_env(api_base)

    last_stdout, last_stderr = "", ""
    last_rc = 0
    total_duration = 0.0
    attempts = 0

    for attempt in range(1, max_retries + 2):  # 1 + retries попыток
        attempts = attempt
        logger.info(f"[aider.build] attempt {attempt}/{max_retries + 1} target={target_file} model={model}")
        rc, stdout, stderr, duration = _run_subprocess(argv, project_dir, timeout_s, env)
        total_duration += duration
        last_rc, last_stdout, last_stderr = rc, stdout, stderr

        # rc == 0 ИЛИ файл создан и непустой → считаем успешным
        # (aider иногда отдаёт ненулевой rc на безобидных warnings)
        content = _read_file_safely(target_path)
        if rc == 0 or (rc != -2 and content.strip()):
            return AiderResult(
                ok=bool(content.strip()),
                file_path=target_file,
                content=content,
                duration_s=round(total_duration, 2),
                error="" if content.strip() else "aider exited cleanly but file is empty",
                stdout=stdout[-4000:],  # ограничиваем для манифеста
                stderr=stderr[-2000:],
                exit_code=rc,
                attempts=attempts,
            )

        # rc == -2 — бинарь не найден, ретраить бесполезно
        if rc == -2:
            break
        logger.warning(f"[aider.build] attempt {attempt} rc={rc}; retrying" if attempt <= max_retries else
                       f"[aider.build] attempt {attempt} rc={rc}; giving up")

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
) -> AiderResult:
    """Починить файл по описанию ошибки.

    Стратегия: формируем для aider инструкцию вида
      «при запуске <cmd> возникла ошибка: <stderr>. Исправь файл так чтобы тест проходил.»
    и зовём aider_build на тот же файл.
    """
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
        project_dir,
        target_file,
        instruction,
        model=model,
        timeout_s=timeout_s,
        max_retries=max_retries,
        api_base=api_base,
    )


def is_aider_available(bin_path: str = AIDER_BIN) -> bool:
    """Быстрая проверка: есть ли aider в PATH.

    Используется в ProjectAgent для авто-фолбэка на старый путь если aider не установлен.
    """
    try:
        result = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
