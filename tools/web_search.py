from __future__ import annotations

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))
