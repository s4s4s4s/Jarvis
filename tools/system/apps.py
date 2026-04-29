# tools/system/apps.py
"""
Управление приложениями Windows для Jarvis.

Инструменты: app.launch, app.kill, app.list, app.active_window
"""
from __future__ import annotations

import subprocess
import sys

_PLATFORM = sys.platform


def launch_app(command: str, args: list[str] | None = None, wait: bool = False) -> dict:
    """
    Запускает приложение.
    command: путь к exe или имя в PATH (например "notepad", "code", "calc")
    Возвращает: {command, pid, waited}
    """
    cmd = [command] + (args or [])
    try:
        if wait:
            result = subprocess.run(cmd, timeout=30, capture_output=True, text=True)
            return {
                "command": command,
                "pid":     None,
                "waited":  True,
                "returncode": result.returncode,
                "stdout":  result.stdout[:2000],
                "stderr":  result.stderr[:500],
            }
        else:
            proc = subprocess.Popen(cmd)
            return {"command": command, "pid": proc.pid, "waited": False}
    except FileNotFoundError:
        raise ValueError(f"Команда не найдена: '{command}'")
    except Exception as e:
        raise RuntimeError(f"Ошибка запуска '{command}': {e}")


def kill_app(name_or_pid: str) -> dict:
    """
    Завершает процесс по имени (.exe) или PID.
    Windows: taskkill /F /IM <name> или /PID <pid>
    Возвращает: {target, killed}
    """
    if _PLATFORM != "win32":
        raise RuntimeError("kill_app поддерживается только на Windows")

    try:
        pid = int(name_or_pid)
        cmd = ["taskkill", "/F", "/PID", str(pid)]
    except ValueError:
        name = name_or_pid if name_or_pid.endswith(".exe") else name_or_pid + ".exe"
        cmd = ["taskkill", "/F", "/IM", name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        killed = result.returncode == 0
        return {"target": name_or_pid, "killed": killed, "output": result.stdout.strip()}
    except Exception as e:
        raise RuntimeError(f"taskkill error: {e}")


def list_processes(filter_name: str = "") -> dict:
    """
    Список запущенных процессов (Windows: tasklist).
    filter_name: фильтрация по имени (регистронезависимо)
    Возвращает: {processes: [{name, pid, memory_kb}]}
    """
    if _PLATFORM != "win32":
        raise RuntimeError("list_processes поддерживается только на Windows")
    result = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, timeout=15,
    )
    processes = []
    for line in result.stdout.splitlines():
        line = line.strip().strip('"')
        if not line:
            continue
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 5:
            continue
        name, pid_s, _, _, mem_s = parts[:5]
        if filter_name and filter_name.lower() not in name.lower():
            continue
        try:
            mem_kb = int(mem_s.replace(" K", "").replace(",", "").replace(".", ""))
        except ValueError:
            mem_kb = 0
        processes.append({"name": name, "pid": pid_s, "memory_kb": mem_kb})
    return {"processes": processes[:100]}


def get_active_window() -> dict:
    """
    Возвращает заголовок активного окна (Windows only).
    Возвращает: {title, hwnd}
    """
    if _PLATFORM != "win32":
        return {"title": "N/A (not Windows)", "hwnd": None}
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return {"title": buf.value, "hwnd": hwnd}
    except Exception as e:
        return {"title": f"error: {e}", "hwnd": None}
