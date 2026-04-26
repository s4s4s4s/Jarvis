import xml.etree.ElementTree as ET
from datetime import date

import requests
from ddgs import DDGS


# ─── ЦБ РФ — фиатные валюты ──────────────────────────────────────────────────

def _cbr_rate(char_code: str) -> str | None:
    try:
        today = date.today().strftime("%d/%m/%Y")
        url = f"https://www.cbr.ru/scripts/XML_daily.asp?date_req={today}"
        resp = requests.get(url, timeout=8)
        resp.encoding = "windows-1251"
        root = ET.fromstring(resp.text)
        for valute in root.findall("Valute"):
            code = valute.find("CharCode").text
            if code == char_code:
                nominal = valute.find("Nominal").text
                value = valute.find("Value").text.replace(",", ".")
                name = valute.find("Name").text
                rate = float(value) / float(nominal)
                return f"{name} ({char_code}): {rate:.2f} руб. (данные ЦБ РФ на {today})"
    except Exception as e:
        print(f"[cbr] Ошибка: {e}")
    return None


# ─── CoinGecko — криптовалюты ─────────────────────────────────────────────────

_CRYPTO_MAP = {
    "биткоин": "bitcoin", "биткоина": "bitcoin", "биткоину": "bitcoin",
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "эфир": "ethereum", "эфириум": "ethereum", "ethereum": "ethereum", "eth": "ethereum",
    "тон": "the-open-network", "tonсoin": "the-open-network", "ton": "the-open-network",
    "солана": "solana", "solana": "solana", "sol": "solana",
    "usdt": "tether", "тезер": "tether", "tether": "tether",
    "bnb": "binancecoin", "бинанс коин": "binancecoin",
}


def _detect_crypto(text: str) -> str | None:
    t = text.lower()
    for keyword, coin_id in _CRYPTO_MAP.items():
        if keyword in t:
            return coin_id
    return None


def _coingecko_rate(coin_id: str) -> str | None:
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        resp = requests.get(
            url,
            params={"ids": coin_id, "vs_currencies": "rub,usd"},
            timeout=8,
        )
        data = resp.json()
        if coin_id not in data:
            return None
        rub = data[coin_id].get("rub")
        usd = data[coin_id].get("usd")
        name = coin_id.replace("-", " ").title()
        return f"{name}: {rub:,.0f} руб. / {usd:,.0f} USD (CoinGecko)"
    except Exception as e:
        print(f"[coingecko] Ошибка: {e}")
    return None


# ─── Фиатные валюты ───────────────────────────────────────────────────────────

_CURRENCY_MAP = {
    "доллар": "USD", "dollar": "USD", "usd": "USD", "бакс": "USD", "баксов": "USD",
    "евро": "EUR", "euro": "EUR", "eur": "EUR",
    "юань": "CNY", "yuan": "CNY", "cny": "CNY", "китайск": "CNY",
    "фунт": "GBP", "pound": "GBP", "gbp": "GBP",
    "франк": "CHF", "franc": "CHF", "chf": "CHF",
    "иена": "JPY", "yen": "JPY", "jpy": "JPY",
}


def _detect_currency(text: str) -> str | None:
    t = text.lower()
    for keyword, code in _CURRENCY_MAP.items():
        if keyword in t:
            return code
    return None


# ─── DuckDuckGo — общий поиск ─────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> str | None:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        parts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            if title or body:
                parts.append(f"{title}: {body}")
        return "\n\n".join(parts) if parts else None
    except Exception as e:
        print(f"[ddg] Ошибка: {e}")
        return None


# ─── публичная функция ────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    # Криптовалюта?
    coin_id = _detect_crypto(query)
    if coin_id:
        result = _coingecko_rate(coin_id)
        if result:
            print(f"[web_search] CoinGecko: {result}")
            return result

    # Фиатная валюта?
    currency_code = _detect_currency(query)
    if currency_code:
        result = _cbr_rate(currency_code)
        if result:
            print(f"[web_search] ЦБ РФ: {result}")
            return result

    # Общий поиск
    result = _ddg_search(query, max_results=max_results)
    if result:
        return result

    return "По запросу ничего не нашёл."