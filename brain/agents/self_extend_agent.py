"""brain/agents/self_extend_agent.py

SelfExtendAgent — Jarvis extends himself.

This is a specialization of CodeDevAgent focused on adding NEW capabilities
to Jarvis itself: new agents, new tools, new routes.

The same pattern is used when Jarvis builds features for external projects —
he learned it on himself first.

Capabilities:
  1. scaffold_agent(name, spec)  — create a new agent file from spec
  2. scaffold_tool(name, spec)   — create a new tool in tools/
  3. register_route(name)        — add route to ROUTER_SYSTEM + ask.py _dispatch
  4. run(query)                  — public entry, called when route=="extend"

Trigger phrases:
  "добавь себе агент <name> который ..."
  "создай инструмент <name> для ..."
  "научись <capability>"
  "add agent <name> that ..."
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

from brain.client import chat, MODEL_HEAVY
from brain.ask import report_progress

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENTS_DIR  = REPO_ROOT / "brain" / "agents"
_TOOLS_DIR   = REPO_ROOT / "tools"
_ASK_PY      = REPO_ROOT / "brain" / "ask.py"
_PROMPTS_PY  = REPO_ROOT / "brain" / "prompts.py"


# ---------------------------------------------------------------------------
# Scaffold agent
# ---------------------------------------------------------------------------

_AGENT_SCAFFOLD_SYSTEM = """\
You are writing a new Python agent module for the Jarvis AI assistant.

The agent must follow these conventions:
  1. Module docstring explaining what the agent does and trigger phrases
  2. A public `run(query: str, history: list[dict] | None = None) -> str` function
  3. Use `from brain.client import chat, MODEL_HEAVY` for LLM calls
  4. Use `from brain.ask import report_progress` for progress updates
  5. Return a human-readable string (markdown OK)
  6. Handle errors gracefully — never let exceptions propagate from run()
  7. Import only stdlib + brain.* + tools.* — no new pip dependencies unless essential

Output the COMPLETE Python file content. No markdown fences, no explanation. Just Python.
"""


def scaffold_agent(name: str, spec: str) -> tuple[Path, str]:
    """
    Generate a new agent file from spec.
    Returns (file_path, content).
    """
    # Read a few existing agents as style examples
    examples = []
    for example_file in ["web_agent.py", "code_agent.py", "plan_agent.py"]:
        p = _AGENTS_DIR / example_file
        if p.exists():
            examples.append(f"=== {example_file} ===\n{p.read_text(encoding='utf-8')[:3000]}")

    messages = [
        {"role": "system", "content": _AGENT_SCAFFOLD_SYSTEM},
        {"role": "user", "content": (
            f"Agent name: {name}\n"
            f"Spec: {spec}\n\n"
            f"Style examples from this codebase:\n\n" + "\n\n".join(examples)
        )},
    ]
    content = chat(MODEL_HEAVY, messages, options={"temperature": 0.1, "num_ctx": 16384})
    content = content.strip()
    # Strip markdown fences if model added them
    if content.startswith("```"):
        content = re.sub(r"^```[\w]*\n", "", content)
        content = re.sub(r"\n```$", "", content)

    # Validate syntax
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise ValueError(f"Generated agent has syntax error: {e}") from e

    file_path = _AGENTS_DIR / f"{name}_agent.py"
    file_path.write_text(content, encoding="utf-8")
    logger.info("[SelfExtend] Created agent: %s", file_path)
    return file_path, content


# ---------------------------------------------------------------------------
# Scaffold tool
# ---------------------------------------------------------------------------

_TOOL_SCAFFOLD_SYSTEM = """\
You are writing a new Python tool module for the Jarvis AI assistant.

Tools are in the tools/ directory and do concrete work (API calls, file I/O, etc).
They are called by tool_agent with a tool name like "weather.current".

Conventions:
  1. Module docstring explaining the tool and its function names
  2. Each public function takes simple arguments and returns a dict or str
  3. All exceptions must be caught and returned as {"error": "...", "ok": False}
  4. Successful results return {"ok": True, "data": ...} or a plain string
  5. No hardcoded API keys — read from environment variables or core/config.py
  6. Functions should be fast (<5s) unless doing file I/O

Output the COMPLETE Python file. No markdown, no explanation. Just Python.
"""


def scaffold_tool(name: str, spec: str) -> tuple[Path, str]:
    """
    Generate a new tool file in tools/.
    Returns (file_path, content).
    """
    examples = []
    for tf in sorted(_TOOLS_DIR.glob("*.py"))[:3]:
        examples.append(f"=== {tf.name} ===\n{tf.read_text(encoding='utf-8')[:2000]}")

    messages = [
        {"role": "system", "content": _TOOL_SCAFFOLD_SYSTEM},
        {"role": "user", "content": (
            f"Tool name: {name}\nSpec: {spec}\n\n"
            + "\n\n".join(examples)
        )},
    ]
    content = chat(MODEL_HEAVY, messages, options={"temperature": 0.1, "num_ctx": 16384})
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[\w]*\n", "", content)
        content = re.sub(r"\n```$", "", content)

    try:
        ast.parse(content)
    except SyntaxError as e:
        raise ValueError(f"Generated tool has syntax error: {e}") from e

    file_path = _TOOLS_DIR / f"{name}.py"
    file_path.write_text(content, encoding="utf-8")
    logger.info("[SelfExtend] Created tool: %s", file_path)
    return file_path, content


# ---------------------------------------------------------------------------
# Register route in ask.py
# ---------------------------------------------------------------------------

def register_route_in_ask(route_name: str, agent_module: str) -> bool:
    """
    Add a new elif branch to _dispatch() in brain/ask.py.
    Returns True if successfully patched.
    """
    if not _ASK_PY.exists():
        logger.warning("[SelfExtend] ask.py not found")
        return False

    source = _ASK_PY.read_text(encoding="utf-8")

    # Don't double-register
    if f'route == "{route_name}"' in source:
        logger.info("[SelfExtend] Route %s already registered", route_name)
        return True

    new_block = (
        f'    if route == "{route_name}":\n'
        f'        from brain.agents.{agent_module} import run as {route_name}_run\n'
        f'        return {route_name}_run(text, history)\n\n'
    )

    # Insert before the final chat fallback
    fallback_marker = "    from brain.agents.chat import run as chat_run"
    if fallback_marker not in source:
        logger.warning("[SelfExtend] Could not find fallback marker in ask.py")
        return False

    patched = source.replace(fallback_marker, new_block + fallback_marker, 1)

    try:
        ast.parse(patched)
    except SyntaxError as e:
        logger.error("[SelfExtend] Patched ask.py has syntax error: %s", e)
        return False

    _ASK_PY.write_text(patched, encoding="utf-8")
    logger.info("[SelfExtend] Registered route '%s' in ask.py", route_name)
    return True


# ---------------------------------------------------------------------------
# Register route in ROUTER_SYSTEM prompt
# ---------------------------------------------------------------------------

def register_route_in_router(route_name: str, description: str, examples: list[str]) -> bool:
    """
    Add new route to ROUTER_SYSTEM in brain/prompts.py.
    Returns True if successfully patched.
    """
    if not _PROMPTS_PY.exists():
        logger.warning("[SelfExtend] prompts.py not found")
        return False

    source = _PROMPTS_PY.read_text(encoding="utf-8")

    if f'"{route_name}"' in source and "route" in source:
        logger.info("[SelfExtend] Route %s already in prompts.py", route_name)
        return True

    examples_str = " / ".join(f'"{e}"' for e in examples[:3])
    new_route_line = f'  - "{route_name}" — {description} (e.g. {examples_str})'

    # Find routes list in ROUTER_SYSTEM and append
    # Look for last route entry pattern
    route_block_pattern = re.compile(
        r'(  - "(?:chat|tool|web|deep|code|plan|test|analyze|extend|develop)"[^\n]*\n)',
        re.MULTILINE,
    )
    matches = list(route_block_pattern.finditer(source))
    if not matches:
        logger.warning("[SelfExtend] Could not find routes block in prompts.py")
        return False

    last_match = matches[-1]
    insert_pos = last_match.end()
    patched = source[:insert_pos] + new_route_line + "\n" + source[insert_pos:]

    try:
        ast.parse(patched)
    except SyntaxError as e:
        logger.error("[SelfExtend] Patched prompts.py has syntax error: %s", e)
        return False

    _PROMPTS_PY.write_text(patched, encoding="utf-8")
    logger.info("[SelfExtend] Added route '%s' to prompts.py", route_name)
    return True


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

_AGENT_PATTERN = re.compile(
    r"(агент|agent)\s+['\"]?([\w_-]+)['\"]?\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_PATTERN = re.compile(
    r"(инструмент|tool)\s+['\"]?([\w_-]+)['\"]?\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_query(query: str) -> tuple[str, str, str]:
    """
    Returns (kind, name, spec).
    kind: "agent" | "tool" | "unknown"
    """
    m = _AGENT_PATTERN.search(query)
    if m:
        return "agent", m.group(2), m.group(3).strip()

    m = _TOOL_PATTERN.search(query)
    if m:
        return "tool", m.group(2), m.group(3).strip()

    # Ask LLM to parse
    return "unknown", "", query


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(query: str, history: list[dict] | None = None) -> str:
    """
    Entry point for SelfExtendAgent.
    Called from _dispatch when route == "extend".
    """
    kind, name, spec = _parse_query(query)

    # If ambiguous, ask LLM to classify
    if kind == "unknown":
        messages = [
            {"role": "system", "content": (
                "Parse the user's intent for extending an AI assistant. "
                "Output JSON: {\"kind\": \"agent\" or \"tool\", \"name\": \"snake_case_name\", \"spec\": \"description\"}"
            )},
            {"role": "user", "content": query},
        ]
        try:
            raw = chat(MODEL_HEAVY, messages, options={"temperature": 0.0, "num_ctx": 4096})
            raw = raw.strip().lstrip("`").rstrip("`")
            data = json.loads(raw)
            kind = data.get("kind", "agent")
            name = data.get("name", "custom")
            spec = data.get("spec", query)
        except Exception:
            kind, name, spec = "agent", "custom", query

    name = re.sub(r"[^\w]", "_", name.lower()).strip("_")

    if kind == "agent":
        report_progress(f"🔨 Генерирую агент `{name}_agent.py`...")
        try:
            file_path, _ = scaffold_agent(name, spec)
        except ValueError as e:
            return f"❌ Ошибка при генерации агента: {e}"

        report_progress(f"🔗 Регистрирую роут `{name}`...")
        route_ok = register_route_in_ask(name, f"{name}_agent")
        prompt_ok = register_route_in_router(
            name,
            spec[:80],
            [f"{name}", f"запусти {name}"],
        )

        status = []
        status.append(f"✅ Создан `{file_path.relative_to(REPO_ROOT)}`")
        status.append(f"{'\u2705' if route_ok else '\u26a0\ufe0f'} Маршрут `{name}` в ask.py {'OK' if route_ok else 'не добавлен'}")
        status.append(f"{'\u2705' if prompt_ok else '\u26a0\ufe0f'} Роут в ROUTER_SYSTEM {'OK' if prompt_ok else 'не добавлен'}")
        status.append(f"\n📌 Теперь можно сказать Ярвису: `{name}: <запрос>`")
        return "\n".join(status)

    elif kind == "tool":
        report_progress(f"🔨 Генерирую инструмент `{name}.py`...")
        try:
            file_path, _ = scaffold_tool(name, spec)
        except ValueError as e:
            return f"❌ Ошибка при генерации инструмента: {e}"

        return (
            f"✅ Создан `{file_path.relative_to(REPO_ROOT)}`\n"
            f"📌 Чтобы использовать, зарегистрируй `{name}.function_name` в tool_agent.py"
        )

    return "❌ Не удалось определить тип расширения. Уточни: агент или инструмент?"
