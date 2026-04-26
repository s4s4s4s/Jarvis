from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

import requests

_CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"


def get_rates() -> dict[str, Any]:
    response = requests.get(_CBR_DAILY_URL, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    rates: dict[str, Any] = {
        "date": root.attrib.get("Date"),
        "base": "RUB",
        "currencies": {},
    }

    for valute in root.findall("Valute"):
        code = (valute.findtext("CharCode") or "").upper()
        nominal = int(valute.findtext("Nominal") or "1")
        name = valute.findtext("Name") or code
        value_text = (valute.findtext("Value") or "0").replace(",", ".")
        value = float(value_text)
        rates["currencies"][code] = {
            "name": name,
            "nominal": nominal,
            "value_rub": value,
            "unit_rate_rub": value / nominal if nominal else None,
        }

    return rates


def convert_currency(amount: float, from_code: str, to_code: str) -> dict[str, Any]:
    rates = get_rates()
    from_code = from_code.upper()
    to_code = to_code.upper()

    if from_code == "RUB":
        rub_amount = amount
    else:
        from_info = rates["currencies"].get(from_code)
        if not from_info:
            raise ValueError(f"Unsupported currency: {from_code}")
        rub_amount = amount * from_info["unit_rate_rub"]

    if to_code == "RUB":
        converted = rub_amount
    else:
        to_info = rates["currencies"].get(to_code)
        if not to_info:
            raise ValueError(f"Unsupported currency: {to_code}")
        converted = rub_amount / to_info["unit_rate_rub"]

    return {
        "amount": amount,
        "from": from_code,
        "to": to_code,
        "result": converted,
        "date": rates["date"],
    }
