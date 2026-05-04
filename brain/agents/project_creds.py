"""
brain/agents/project_creds.py — управление учётными данными для ProjectAgent.

Принцип: каждый проект хранит свои секреты в .env внутри папки проекта.
Jarvis НИКОГДА не пишет токены/ключи в core/config.py.
5 проектов = 5 изолированных .env.

Workflow:
  1. INTAKE — PROJECT_INTAKE_SYSTEM извлекает spec.credentials[] из запроса.
  2. project.run() вызывает check_and_request_creds() после intake.
  3. Если нужны данные — возвращается текстовый список (не TTS — токены длинные).
  4. Пользователь вводит KEY=VALUE в следующем сообщении.
  5. ask.py обнаруживает has_pending_creds() + looks_like_creds() → provide_credentials().
  6. provide_credentials() записывает .env в папку проекта → project.resume(slug).
  7. Coder читает os.environ + load_dotenv() — подхватывает .env автоматически.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _data_dir() -> Path:
    """DATA_DIR из core.paths как Path."""
    from core.paths import DATA_DIR
    return Path(DATA_DIR)


def pending_creds_path() -> Path:
    """Путь к файлу ожидания учётных данных."""
    return _data_dir() / "projects" / "_pending_creds.json"


def save_pending_creds(slug: str, spec: dict) -> None:
    """Сохранить состояние ожидания учётных данных."""
    pending_file = pending_creds_path()
    try:
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        pending_file.write_text(
            json.dumps({"slug": slug, "spec": spec}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"[project_creds] pending saved for slug={slug}")
    except Exception as e:
        logger.warning(f"[project_creds] failed to save pending: {e}")


def load_pending_creds() -> dict | None:
    """Загрузить ожидающий проект. None если нет ожидающего."""
    pending_file = pending_creds_path()
    if not pending_file.exists():
        return None
    try:
        return json.loads(pending_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[project_creds] failed to load pending: {e}")
        return None


def clear_pending_creds() -> None:
    """Удалить файл ожидания."""
    try:
        pending_creds_path().unlink(missing_ok=True)
    except Exception:
        pass


def format_creds_request(spec: dict, creds: list) -> str:
    """Форматирует список нужных учётных данных для вывода пользователю.

    Возвращает ТЕКСТ (не голос) — токены слишком длинные для TTS.
    """
    title = spec.get("title") or "проект"
    lines = [f"Для проекта «{title}» нужны следующие данные:\n"]
    for c in creds:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or ""
        desc = c.get("description") or ""
        how = c.get("how_to_get") or ""
        line = f"  • {name}"
        if desc:
            line += f" — {desc}"
        lines.append(line)
        if how:
            lines.append(f"    Как получить: {how}")
    lines.append("")
    lines.append("Введите значения ответным сообщением в формате:")
    for c in creds:
        if isinstance(c, dict) and c.get("name"):
            lines.append(f"  {c['name']}=ваше_значение")
    return "\n".join(lines)


def has_pending_creds() -> bool:
    """Есть ли проект, ожидающий учётных данных."""
    return pending_creds_path().exists()


def looks_like_creds(text: str) -> bool:
    """Быстрая проверка — похож ли текст на ввод KEY=VALUE."""
    return bool(re.search(r"[A-Z_][A-Z0-9_]{2,}\s*=\s*\S", text or ""))


def check_and_request_creds(slug: str, spec: dict) -> str | None:
    """Проверить нужны ли учётные данные для проекта.

    Вызывается из project.run() после intake до architect.
    
    Returns:
        str: форматированный запрос нужных данных — если нужны и ещё не предоставлены
        None: если данные не нужны или уже есть в .env — продолжать сборку
    """
    creds = [c for c in (spec.get("credentials") or [])
             if isinstance(c, dict) and c.get("name")]
    if not creds:
        return None

    # .env уже есть (resume-сценарий или проверка) — не прерываем
    try:
        from tools.projects import project_dir
        env_path = project_dir(slug) / ".env"
        if env_path.exists():
            return None
    except Exception:
        pass

    save_pending_creds(slug, spec)
    return format_creds_request(spec, creds)


def provide_credentials(text: str) -> str:
    """Принять KEY=VALUE от пользователя и продолжить ожидающий проект.

    Вызывается из ask.py когда has_pending_creds() и looks_like_creds().
    """
    pending = load_pending_creds()
    if not pending:
        return "Нет проекта, ожидающего учётных данных."

    slug = pending.get("slug")
    if not slug:
        clear_pending_creds()
        return "Повреждённые данные ожидающего проекта."

    # Разбираем KEY=VALUE пары
    env_lines: list[str] = []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if "=" in line and re.match(r"^[A-Z_][A-Z0-9_]*\s*=", line):
            key, val = line.split("=", 1)
            env_lines.append(f"{key.strip()}={val.strip()}")

    if not env_lines:
        return (
            "Не нашёл данных в формате KEY=VALUE.\n"
            "Введите, например:\n"
            "  TELEGRAM_BOT_TOKEN=7123456789:AAHdq..."
        )

    # Записываем .env в папку проекта
    try:
        from tools.projects import project_dir
        env_path = project_dir(slug) / ".env"
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        keys_saved = [ln.split("=")[0] for ln in env_lines]
        logger.info(f"[project_creds] .env written for slug={slug}, keys={keys_saved}")
    except Exception as e:
        return f"Не удалось сохранить учётные данные: {e}"

    # Убираем pending-метку
    clear_pending_creds()

    # Продолжаем проект с фазы architect
    try:
        from brain.agents.project import resume
        return resume(slug)
    except Exception as e:
        return f"Учётные данные сохранены, но возобновить проект не удалось: {e}"
