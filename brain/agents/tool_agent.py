from __future__ import annotations

from tools.registry import call_tool


def tool_agent(tool: str, tool_args: dict):
    return call_tool(tool, tool_args)
