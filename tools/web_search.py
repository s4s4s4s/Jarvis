from __future__ import annotations

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # fallback for old installs

_SEARCH_TIMEOUT = 10  # секунд — защита от зависания DDG


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search DuckDuckGo and return results as a formatted string
    ready to be placed directly into an LLM prompt.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, timeout=_SEARCH_TIMEOUT))
    except Exception as e:
        return f"Поиск недоступен: {e}"

    if not results:
        return "Результатов не найдено."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        body  = r.get("body", "").strip()
        href  = r.get("href", "").strip()
        lines.append(f"[{i}] {title}\n{body}\nИсточник: {href}")

    return "\n\n".join(lines)
