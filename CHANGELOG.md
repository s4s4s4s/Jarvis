# Changelog

## [beta-1.1] — 2026-04-27

### Добавлено
- **`auditor.run`** — голосовой аудит кода через `tool_agent`. Фразы-триггеры: "проверь код", "найди баги", "проаудируй"
- **`tools/registry.py`** — `auditor.run` зарегистрирован в `_TOOL_MAP`
- **`brain/prompts.py`** — `auditor.run` добавлен в `ROUTER_SYSTEM` (инструмент #10)

### Изменено
- **`dev/auditor.py`** — перешёл с `requests`/llama-server на `brain.client.chat()` (Ollama). Нет конфликта по VRAM с Whisper
- **`dev/auditor.py`** — добавлены исключения в `SYSTEM_PROMPT`: `_TOOL_MAP` регистри, fallback-константы, комментарии — не флагаются как HARDCODED_STRING
- **`core/config.py`** — `EDGE_RATE` `+50%` → `+30%` (голос медленнее)

### Исправлено (audit 7)
- **`brain/ask.py`** — хардкод строки `"get_answer"` и `"получение ответа"` вынесены в `_CTX_TIMEOUT` / `_TOOL_TIMEOUT`
- **`brain/ask.py`** — хардкод fallback `"Сэр, инструмент вернул ошибку"` → константа `_FALLBACK_ERROR`

## [beta-1.0] — 2026-04-20

- Базовая сборка: STT (Whisper large-v3), TTS (Edge TTS), Ollama-роутер
- Инструменты: weather, crypto, currency, time, timer
- Долгосрочная память, wake-word детектор, перебивание TTS
