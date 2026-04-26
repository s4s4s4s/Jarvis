from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.crypto import get_crypto_price, search_coin
from tools.currency import convert_currency, get_rates
from tools.time_tool import get_time
from tools.weather import get_weather
from tools.timer import set_timer, list_timers, cancel_timer
from dev.auditor import AuditorAgent


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""


def _call_weather(location: str, language: str = "ru") -> Any:
    return get_weather(location=location, language=language)


def _call_crypto_search(query: str) -> Any:
    return search_coin(query=query)


def _call_crypto_price(ids: list[str], vs_currency: str = "usd") -> Any:
    return get_crypto_price(ids=ids, vs_currency=vs_currency)


def _call_currency_rates() -> Any:
    return get_rates()


def _call_currency_convert(amount: float, from_code: str, to_code: str) -> Any:
    return convert_currency(amount=amount, from_code=from_code, to_code=to_code)


def _call_time() -> Any:
    return get_time()


def _call_timer_set(seconds: int, label: str = "таймер") -> Any:
    return set_timer(seconds=seconds, label=label)


def _call_timer_list() -> Any:
    return list_timers()


def _call_timer_cancel(timer_id: str) -> Any:
    ok = cancel_timer(timer_id)
    return {"cancelled": ok, "id": timer_id}


_DEFAULT_AUDIT_FILES = [
    "brain/ask.py",
    "brain/agents/chat.py",
    "brain/agents/tool_agent.py",
    "tools/registry.py",
]


def _call_auditor(files: list[str] | None = None) -> str:
    """Run AuditorAgent on project files and return a plain-text summary for LLM."""
    targets = files or _DEFAULT_AUDIT_FILES
    agent = AuditorAgent()
    findings = agent.audit(targets)

    if not findings:
        return "Находок не обнаружено."

    confirmed   = [f for f in findings if f.status == "confirmed"]
    needs_review = [f for f in findings if f.status == "needs_review"]
    rejected    = [f for f in findings if f.status == "rejected"]

    lines = [
        f"Аудит {len(targets)} файл(ов): {len(findings)} находок, "
        f"{len(confirmed)} подтверждено, "
        f"{len(needs_review)} на проверке, "
        f"{len(rejected)} отклонено.",
        "",
    ]

    for f in confirmed:
        lines.append(f"[ПОДТВЕРЖДЕНО] {f.file}:{f.line} ({f.type}, conf={f.confidence:.2f})")
        lines.append(f"  Проблема: {f.description}")
        lines.append(f"  Решение: {f.suggestion}")
        lines.append("")

    for f in needs_review:
        lines.append(f"[ПРОВЕРКА] {f.file}:{f.line} ({f.type}, conf={f.confidence:.2f})")
        lines.append(f"  Проблема: {f.description}")
        lines.append(f"  Причина: {f.reject_reason}")
        lines.append("")

    return "\n".join(lines)


_TOOL_MAP: dict[str, Any] = {
    "weather":          _call_weather,
    "crypto.search":    _call_crypto_search,
    "crypto.price":     _call_crypto_price,
    "currency.rates":   _call_currency_rates,
    "currency.convert": _call_currency_convert,
    "time":             _call_time,
    "timer.set":        _call_timer_set,
    "timer.list":       _call_timer_list,
    "timer.cancel":     _call_timer_cancel,
    "auditor.run":      _call_auditor,
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
