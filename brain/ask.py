from __future__ import annotations

from brain.agents.tool_agent import tool_agent


def ask(route: str, message: str, tool: str | None = None, tool_args: dict | None = None):
    if route == "tool" and tool:
        return tool_agent(tool, tool_args or {})
    raise NotImplementedError("Chat route handling is not implemented here")
