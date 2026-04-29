# Changelog

## [beta-1.2] — 2026-04-29

### Добавлено
- **`dev/self_test_agent.py`** — исправлен сломанный импорт `_route`→`_route_llm`, добавлена разбивка по источникам (`embed`/`llm`/`error`)
- **`dev/nightly_self_heal.py`** — замкнутый self-heal цикл: baseline → learning → post-test → автооткат если pass rate упал
- **`dev/schedule_nightly.ps1`** — Task Scheduler скрипт, запуск каждую ночь в 03:00
- **`dev/watch_pass_rate.py`** — мониторинг pass rate + alert в `logs/alerts.jsonl`
- **`tools/system/`** — папка системных инструментов (Level 2):
  - `files.py` — file.read / file.write / file.list / file.search / file.delete
  - `apps.py` — app.launch / app.kill / app.list / app.active_window
  - `browser.py` — browser.open / browser.search / browser.get_text
  - `clipboard.py` — clipboard.get / clipboard.set
- **`tools/registry.py`** — все системные инструменты зарегистрированы
- **`brain/agents/planner.py`** — PlannerAgent: LLM декомпозиция → пошаговое выполнение инструментов → синтез
- **`brain/ask.py`** — добавлен маршрут `route="plan"` → `planner.py`
- **`data/route_examples.jsonl`** — добавлены примеры для plan/system-инструментов

## [beta-1.1] — 2026-04-27

### Добавлено
- **`auditor.run`** — голосовой аудит кода через `tool_agent`
- **`tools/registry.py`** — `auditor.run` зарегистрирован
- **`brain/prompts.py`** — `auditor.run` добавлен в `ROUTER_SYSTEM`

### Изменено
- **`dev/auditor.py`** — перешёл с `requests` на `brain.client.chat()` (Ollama)
- **`core/config.py`** — `EDGE_RATE` `+50%` → `+30%`

### Исправлено
- **`brain/ask.py`** — хардкод строк вынесен в константы `_CTX_TIMEOUT` / `_TOOL_TIMEOUT` / `_FALLBACK_ERROR`

## [beta-1.0] — 2026-04-20

- Базовая сборка: STT (Whisper large-v3), TTS (Edge TTS), Ollama-роутер
- Инструменты: weather, crypto, currency, time, timer
- Долгосрочная память, wake-word детектор, перебивание TTS
