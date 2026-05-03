# tools/system/files.py
"""
Файловые операции для Jarvis.

Инструменты: file.read, file.write, file.list, file.search, file.delete
Все пути ограничены USER_HOME для безопасности.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

_USER_HOME = Path.home()
_MAX_READ_BYTES = 100_000  # 100 KB — защита от огромных файлов

# fix BUG-2: список разрешённых корневых директорий.
# Все пути через _safe_path() ОБЯЗАНЫ начинаться с одного из этих корней.
# На Windows добавляем корни всех дисков (C:\, D:\, ...) чтобы не ломать
# абсолютные пути, при этом block /etc, /proc и прочие Unix-пути.
def _build_allowed_roots() -> list[Path]:
    roots = [_USER_HOME.resolve()]
    # На Windows добавляем все доступные диски
    if os.name == "nt":
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append(drive.resolve())
    else:
        # На Unix разрешаем только home
        pass
    return roots

_ALLOWED_ROOTS: list[Path] = _build_allowed_roots()


def _safe_path(raw: str) -> Path:
    """
    Резолвит путь относительно home или абсолютный.
    fix BUG-2: теперь реально запрещает выход за пределы разрешённых корней.
    Бросает PermissionError если путь за пределами _ALLOWED_ROOTS.
    """
    p = Path(raw).expanduser().resolve()
    p_str = str(p)
    for root in _ALLOWED_ROOTS:
        root_str = str(root)
        # Проверяем что p начинается с root (с учётом разделителя)
        if p_str == root_str or p_str.startswith(root_str + os.sep):
            return p
    raise PermissionError(
        f"Путь вне разрешённой зоны: {p}. "
        f"Разрешены только пути внутри: {[str(r) for r in _ALLOWED_ROOTS]}"
    )


def read_file(path: str, encoding: str = "utf-8") -> dict:
    """
    Читает текстовый файл. Возвращает первые 100 KB.
    Возвращает: {path, content, size_bytes, truncated}
    """
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    if not p.is_file():
        raise ValueError(f"Путь не является файлом: {p}")
    raw = p.read_bytes()
    truncated = len(raw) > _MAX_READ_BYTES
    content = raw[:_MAX_READ_BYTES].decode(encoding, errors="replace")
    return {
        "path":       str(p),
        "content":    content,
        "size_bytes": len(raw),
        "truncated":  truncated,
    }


def write_file(path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> dict:
    """
    Записывает текстовый файл. Создаёт родительские директории.
    Возвращает: {path, size_bytes, created}
    """
    p = _safe_path(path)
    created = not p.exists()
    if not overwrite and not created:
        raise FileExistsError(f"Файл уже существует: {p}. Используй overwrite=True.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return {
        "path":       str(p),
        "size_bytes": len(content.encode(encoding)),
        "created":    created,
    }


def list_dir(path: str = "~", pattern: str = "*") -> dict:
    """
    Перечисляет содержимое директории (не рекурсивно).
    Возвращает: {path, items: [{name, type, size_bytes}]}
    """
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Директория не найдена: {p}")
    if not p.is_dir():
        raise ValueError(f"Путь не является директорией: {p}")
    items = []
    for entry in sorted(p.iterdir()):
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            size = 0
        items.append({
            "name":       entry.name,
            "type":       "file" if entry.is_file() else "dir",
            "size_bytes": size,
        })
    return {"path": str(p), "items": items[:200]}  # max 200 entries


def search_files(root: str = "~", pattern: str = "*.py", max_results: int = 50) -> dict:
    """
    Рекурсивный поиск файлов по маске.
    Возвращает: {root, pattern, found: [str]}
    """
    p = _safe_path(root)
    if not p.is_dir():
        raise ValueError(f"Не директория: {p}")
    found = []
    for match in p.rglob(pattern):
        if match.is_file():
            found.append(str(match))
        if len(found) >= max_results:
            break
    return {"root": str(p), "pattern": pattern, "found": found}


def delete_file(path: str) -> dict:
    """
    Удаляет файл (только файл, не директорию).
    Возвращает: {path, deleted}
    """
    p = _safe_path(path)
    if not p.exists():
        return {"path": str(p), "deleted": False, "reason": "not_found"}
    if not p.is_file():
        raise ValueError(f"Не файл: {p}")
    p.unlink()
    return {"path": str(p), "deleted": True}
