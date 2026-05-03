from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.crypto import get_crypto_price, search_coin
from tools.currency import convert_currency, get_rates
from tools.time_tool import get_time
from tools.weather import get_weather
from tools.timer import set_timer, list_timers, cancel_timer
from tools.web_search import web_search

# Системные инструменты (Level 2)
from tools.system.files import read_file, write_file, list_dir, search_files, delete_file
from tools.system.apps import launch_app, kill_app, list_processes, get_active_window
from tools.system.browser import open_url, search_in_browser, get_page_text
from tools.system.clipboard import get_clipboard, set_clipboard

# Векторная память
from tools.memory import add_fact, search_memory, get_all_facts

# AuditorAgent — опциональная зависимость (dev/ может отсутствовать)
try:
    from dev.auditor import AuditorAgent as _AuditorAgent
    _AUDITOR_AVAILABLE = True
except Exception:
    _AuditorAgent = None  # type: ignore[assignment,misc]
    _AUDITOR_AVAILABLE = False


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""


# ── weather ──────────────────────────────────────────────────────────────────
default_location = "Moscow"

def _call_weather(location: str = default_location, language: str = "ru") -> Any:
    return get_weather(location=location, language=language)

# ── crypto ──────────────────────────────────────────────────────────────────
def _call_crypto_search(query: str) -> Any:
    return search_coin(query=query)

def _call_crypto_price(ids: list[str], vs_currency: str = "usd") -> Any:
    return get_crypto_price(ids=ids, vs_currency=vs_currency)

# ── currency ──────────────────────────────────────────────────────────────────
def _call_currency_rates() -> Any:
    return get_rates()

def _call_currency_convert(amount: float, from_code: str, to_code: str) -> Any:
    return convert_currency(amount=amount, from_code=from_code, to_code=to_code)

# ── time / timer ─────────────────────────────────────────────────────────────────
def _call_time() -> Any:
    return get_time()

def _call_timer_set(seconds: int, label: str = "таймер") -> Any:
    return set_timer(seconds=seconds, label=label)

def _call_timer_list() -> Any:
    return list_timers()

def _call_timer_cancel(timer_id: str) -> Any:
    ok = cancel_timer(timer_id)
    return {"cancelled": ok, "id": timer_id}

# ── web search ────────────────────────────────────────────────────────────────
def _call_web_search(query: str, max_results: int = 5) -> Any:
    return web_search(query=query, max_results=max_results)

# ── auditor ────────────────────────────────────────────────────────────────────────
_DEFAULT_AUDIT_FILES = [
    "brain/ask.py",
    "brain/agents/chat.py",
    "brain/agents/tool_agent.py",
    "tools/registry.py",
]

def _call_auditor(files: list[str] | None = None) -> str:
    if not _AUDITOR_AVAILABLE or _AuditorAgent is None:
        return "AuditorAgent недоступен (dev/auditor.py не найден)."
    targets = files or _DEFAULT_AUDIT_FILES
    agent = _AuditorAgent()
    findings = agent.audit(targets)
    if not findings:
        return "Находок не обнаружено."
    confirmed    = [f for f in findings if f.status == "confirmed"]
    needs_review = [f for f in findings if f.status == "needs_review"]
    lines = [
        f"Аудит {len(targets)} файл(ов): {len(findings)} находок, "
        f"{len(confirmed)} подтверждено.", "",
    ]
    for f in confirmed:
        lines += [
            f"[ПОДТВЕРЖДЕНО] {f.file}:{f.line} ({f.type})",
            f"  {f.description}", f"  Решение: {f.suggestion}", "",
        ]
    for f in needs_review:
        lines += [
            f"[ПРОВЕРКА] {f.file}:{f.line} — {f.description}", "",
        ]
    return "\n".join(lines)

# ── system: files ────────────────────────────────────────────────────────────────────
def _call_file_read(path: str) -> Any:
    return read_file(path)

def _call_file_write(path: str, content: str, overwrite: bool = True) -> Any:
    return write_file(path, content, overwrite=overwrite)

def _call_file_list(path: str = "~", pattern: str = "*") -> Any:
    return list_dir(path, pattern)

def _call_file_search(root: str = "~", pattern: str = "*.py") -> Any:
    return search_files(root, pattern)

def _call_file_delete(path: str) -> Any:
    return delete_file(path)

# ── system: apps ────────────────────────────────────────────────────────────────────────
def _call_app_launch(command: str, args: list[str] | None = None, wait: bool = False) -> Any:
    return launch_app(command, args, wait)

def _call_app_kill(name_or_pid: str) -> Any:
    return kill_app(name_or_pid)

def _call_app_list(filter_name: str = "") -> Any:
    return list_processes(filter_name)

def _call_app_active_window() -> Any:
    return get_active_window()

# ── system: browser ─────────────────────────────────────────────────────────────────────
def _call_browser_open(url: str, browser: str = "default") -> Any:
    return open_url(url, browser)

def _call_browser_search(query: str, engine: str = "google") -> Any:
    return search_in_browser(query, engine)

def _call_browser_get_text(url: str) -> Any:
    return get_page_text(url)

# ── system: clipboard ────────────────────────────────────────────────────────────────────
def _call_clipboard_get() -> Any:
    return get_clipboard()

def _call_clipboard_set(text: str) -> Any:
    return set_clipboard(text)

# ── memory ─────────────────────────────────────────────────────────────────────────────────
def _call_memory_search(query: str, n_results: int = 5) -> Any:
    results = search_memory(query=query, n_results=n_results)
    if not results:
        return "Факты не найдены."
    lines = [f"[{r['score']:.2f}] {r['fact']}" for r in results]
    return "\n".join(lines)

def _call_memory_add(fact: str, category: str = "общее", source: str = "") -> Any:
    ok = add_fact(fact=fact, category=category, source=source)
    return {"added": ok, "fact": fact}

def _call_memory_list(n: int = 20) -> Any:
    facts = get_all_facts()
    if not facts:
        return "Память пуста."
    recent = facts[-n:]
    lines  = [f"{i+1}. [{f['category']}] {f['fact']}" for i, f in enumerate(recent)]
    return "\n".join(lines)


# ── Главная таблица инструментов ───────────────────────────────────────────────────────────────
_TOOL_MAP: dict[str, Any] = {
    # Существующие
    "weather":            _call_weather,
    "crypto.search":      _call_crypto_search,
    "crypto.price":       _call_crypto_price,
    "currency.rates":     _call_currency_rates,
    "currency.convert":   _call_currency_convert,
    "time":               _call_time,
    "timer.set":          _call_timer_set,
    "timer.list":         _call_timer_list,
    "timer.cancel":       _call_timer_cancel,
    "web.search":         _call_web_search,
    "auditor.run":        _call_auditor,
    # Файлы (Level 2)
    "file.read":          _call_file_read,
    "file.write":         _call_file_write,
    "file.list":          _call_file_list,
    "file.search":        _call_file_search,
    "file.delete":        _call_file_delete,
    # Приложения (Level 2)
    "app.launch":         _call_app_launch,
    "app.kill":           _call_app_kill,
    "app.list":           _call_app_list,
    "app.active_window":  _call_app_active_window,
    # Браузер (Level 2)
    "browser.open":       _call_browser_open,
    "browser.search":     _call_browser_search,
    "browser.get_text":   _call_browser_get_text,
    # Буфер обмена (Level 2)
    "clipboard.get":      _call_clipboard_get,
    "clipboard.set":      _call_clipboard_set,
    # Векторная память (Level 2)
    "memory.search":      _call_memory_search,
    "memory.add":         _call_memory_add,
    "memory.list":        _call_memory_list,
}


def list_tools() -> list[str]:
    return list(_TOOL_MAP.keys())


def call_tool(name: str, args: dict[str, Any] | None = None) -> ToolResult:
    args = args or {}
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return ToolResult(ok=False, error=f"Unknown tool: {name}. Available: {list_tools()}")
    try:
        result = fn(**args)
        return ToolResult(ok=True, data=result)
    except TypeError as e:
        return ToolResult(ok=False, error=f"Bad arguments for tool '{name}': {e}")
    except ValueError as e:
        return ToolResult(ok=False, error=str(e))
    except Exception as e:
        return ToolResult(ok=False, error=f"Tool '{name}' failed: {e}")
