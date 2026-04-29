# tools/system/browser.py
"""
Управление браузером для Jarvis через subprocess + webbrowser.

Лёгкая реализация без selenium (нет зависимостей):
  browser.open  — открыть URL
  browser.search — поиск в браузере

Полная реализация (Спринт 2.1+) — через playwright:
  browser.click, browser.type, browser.screenshot, browser.get_text
"""
from __future__ import annotations

import subprocess
import sys
import webbrowser

_PLATFORM = sys.platform


def open_url(url: str, browser: str = "default") -> dict:
    """
    Открывает URL в браузере.
    browser: "default" | "chrome" | "firefox" | "edge"
    Возвращает: {url, opened}
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        if browser == "default":
            webbrowser.open(url)
        elif browser in ("chrome", "google-chrome"):
            if _PLATFORM == "win32":
                subprocess.Popen(["start", "chrome", url], shell=True)
            else:
                subprocess.Popen(["google-chrome", url])
        elif browser == "edge":
            if _PLATFORM == "win32":
                subprocess.Popen(["start", "msedge", url], shell=True)
        elif browser == "firefox":
            subprocess.Popen(["firefox", url])
        else:
            webbrowser.open(url)
        return {"url": url, "opened": True, "browser": browser}
    except Exception as e:
        raise RuntimeError(f"Ошибка открытия браузера: {e}")


def search_in_browser(query: str, engine: str = "google") -> dict:
    """
    Открывает поисковый запрос в браузере.
    engine: "google" | "yandex" | "bing" | "duckduckgo"
    Возвращает: {query, url, opened}
    """
    import urllib.parse
    q = urllib.parse.quote_plus(query)
    engines = {
        "google":     f"https://www.google.com/search?q={q}",
        "yandex":     f"https://yandex.ru/search/?text={q}",
        "bing":       f"https://www.bing.com/search?q={q}",
        "duckduckgo": f"https://duckduckgo.com/?q={q}",
    }
    url = engines.get(engine, engines["google"])
    result = open_url(url)
    result["query"] = query
    result["engine"] = engine
    return result


def get_page_text(url: str, timeout: int = 15) -> dict:
    """
    Получает текстовое содержимое страницы через requests + BeautifulSoup.
    Без JavaScript (статичные страницы).
    Возвращает: {url, text, title, status_code}
    """
    try:
        import requests
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._skip = False
                self.texts: list[str] = []
                self.title = ""
                self._in_title = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True
                if tag == "title":
                    self._in_title = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False
                if tag == "title":
                    self._in_title = False

            def handle_data(self, data):
                if self._in_title:
                    self.title += data
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self.texts.append(stripped)

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        parser = _TextExtractor()
        parser.feed(resp.text)
        text = " ".join(parser.texts)[:8000]  # max 8KB текста
        return {
            "url":         url,
            "title":       parser.title.strip(),
            "text":        text,
            "status_code": resp.status_code,
        }
    except ImportError:
        raise RuntimeError("requests не установлен. pip install requests")
    except Exception as e:
        raise RuntimeError(f"Ошибка получения страницы '{url}': {e}")
