from __future__ import annotations

from tools.registry import call_tool, ToolResult


def tool_agent(tool: str, tool_args: dict) -> ToolResult:
    """Dispatch a named tool call, returning a structured ToolResult."""
    return call_tool(tool, tool_args)
