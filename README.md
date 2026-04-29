# Jarvis

Локальный голосовой ассистент на Python 3.11 (Windows 11, RTX 4090).  
Полностью офлайн-LLM через Ollama, Edge TTS для голоса, faster-whisper для распознавания речи.

---

## Стек

| Слой | Технология |
|---|---|
| STT | faster-whisper `large-v3` + Silero VAD |
| TTS | edge-tts `ru-RU-DmitryNeural` + pygame mixer (streaming по предложениям) |
| LLM | Ollama SDK (локально) |
| GUI | PySide6 |
| Аудио | sounddevice + numpy, tap-система через AudioCore |
| Поиск | DuckDuckGo (`duckduckgo-search`) |
| Память | JSON-файл `C:\Jarvis\data\memory.json`, async извлечение фактов через LLM |

### Модели Ollama

| Роль | Модель |
|---|---|
| Router | `qwen2.5:14b-instruct-q4_K_M` |
| Fast (chat, tool format) | `llama3.1:8b-instruct-q5_K_M` |
| Heavy (deep, memory extract) | `qwen2.5:32b-instruct-q4_K_M` |

---

## Архитектура

```
app.py
  └── voice/assistant.py          # главный цикл (wake → listen → think → speak)
        ├── voice/audio_core.py   # единый mic-поток, tap-система для TTS interrupt
        ├── voice/wake.py         # wake-word детектор (WAKE_PHRASES из config)
        ├── voice/turn.py         # TurnManager: VAD + сбор utterance + timeout
        ├── voice/stt.py          # faster-whisper transcribe(audio) → str
        └── voice/tts.py          # Edge TTS streaming + interrupt-listener

brain/ask.py                      # ask_llm(text) → AskResult(filler, Future[answer])
  ├── brain/client.py             # dual-backend: Ollama (serial) / llama-server (parallel)
  ├── brain/router.py             # parse_router_response() — legacy, не используется активно
  ├── brain/prompts.py            # ROUTER_SYSTEM, CHAT_SYSTEM, DEEP_SYSTEM,
  │                               # MEMORY_SYSTEM, TOOL_FORMAT_SYSTEM, WEB_SYSTEM
  ├── brain/history.py            # история диалога: snapshot / append / clear
  ├── brain/logger.py             # router.jsonl лог каждого роута
  ├── brain/llama_server.py       # LlamaServerManager — старт/стоп llama-server.exe
  └── brain/agents/
        ├── executor.py           # async Executor: параллельные + серийные задачи
        ├── planner.py            # PlannerAgent — создаёт план задач из запроса
        ├── types.py              # датаклассы Task
        ├── chat.py               # chat-агент + get_memory_context() в каждый turn
        ├── deep.py               # deep-агент (развёрнутые ответы)
        ├── memory_agent.py       # отвечает на вопросы о пользователе
        ├── web_agent.py          # DuckDuckGo → LLM синтез
        └── tool_agent.py         # вызывает tools/registry.call_tool()

tools/
  ├── registry.py                 # call_tool(name, args) → ToolResult(ok, data, error)
  ├── weather.py                  # Open-Meteo + геокодинг, TTL-кэш 10 мин
  ├── crypto.py                   # CoinGecko search + markets
  ├── currency.py                 # ЦБ РФ XML курсы, промежуточная конвертация через RUB
  ├── time_tool.py                # datetime московского времени (ZoneInfo)
  ├── timer.py                    # countdown таймеры: set/list/cancel, callback → say()
  ├── web_search.py               # DuckDuckGo → отформатированная строка
  └── memory.py                   # долгосрочная память, extract_and_save_async()

core/
  ├── config.py                   # ВСЕ настройки: модели, аудио, VAD, wake, TTS, память
  └── paths.py                    # ВСЕ пути: ROOT, LOGS_DIR, MEMORY_PATH, TTS_CHUNKS
                                  # ensure_dirs() вызывается один раз в app.py

ui/
  ├── main_window.py
  └── bridge.py

assets/                           # activate.wav, deactivate.wav, reference.wav
logs/                             # router.jsonl (в .gitignore)
data/                             # memory.json  (в .gitignore)
```

---

## Маршруты роутера

Роутер (`brain/ask.py → _route()`) возвращает JSON со схемой:
```json
{
  "route": "chat | tool | web | deep | memory",
  "tool": "tool_name | null",
  "tool_args": {},
  "confidence": 0.95,
  "filler": "Сейчас посмотрю...",
  "reason": "краткое пояснение"
}
```

| Маршрут | Когда используется |
|---|---|
| `chat` | Обычный разговор, общие знания |
| `tool` | Живые данные (погода, крипто, курсы, время, таймер) |
| `web` | Актуальные события, поиск в интернете |
| `deep` | Развёрнутый анализ, длинный ответ |
| `memory` | Вопросы о пользователе («как меня зовут», «что ты знаешь обо мне») |

---

## Инструменты (tools/registry.py)

| Имя | Описание | Аргументы |
|---|---|---|
| `weather` | Текущая погода по городу | `location`, `language?` |
| `crypto.search` | Поиск криптовалюты по тексту | `query` |
| `crypto.price` | Цена по CoinGecko id | `ids[]`, `vs_currency?` |
| `currency.rates` | Все курсы ЦБ РФ | — |
| `currency.convert` | Конвертация суммы | `amount`, `from_code`, `to_code` |
| `time` | Текущее московское время | — |
| `timer.set` | Поставить таймер | `seconds`, `label?` |
| `timer.list` | Список активных таймеров | — |
| `timer.cancel` | Отменить таймер | `timer_id` |

---

## Голосовые команды (без LLM)

| Фраза | Действие |
|---|---|
| `очисти историю` / `забудь всё` | Очищает `brain/history` |
| `перейди в пассивный режим` / `пока` | Wake-word режим ожидания |
| `перейди в активный режим` | Постоянное прослушивание |

---

## Установка и запуск

### Быстрый старт

```powershell
# 1. Клонировать репозиторий
git clone https://github.com/s4s4s4s/Jarvis.git C:\Jarvis
cd C:\Jarvis

# 2. Переключиться на рабочую ветку
git checkout feature/planner-agent

# 3. Создать venv и поставить зависимости
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 4. Убедиться что Ollama запущена
ollama list

# 5. Запустить Jarvis
python app.py
```

### Запуск агентного пайплайна (PlannerAgent + Executor)

```powershell
cd C:\Jarvis
.venv\Scripts\activate

# Параллельный режим (запускает llama-server, нужен C:\llama-server\llama-server.exe)
python -m brain.agents.executor "напиши функцию сортировки пузырьком и функцию бинарного поиска"

# Без параллелизма — только Ollama
python -m brain.agents.executor --no-parallel "твой запрос"
```

### deploy.ps1 (альтернатива)

```powershell
# Первый запуск — создаёт venv, ставит зависимости, пушит и запускает
.\deploy.ps1 -msg "init"

# Обычный запуск без установки зависимостей
.\deploy.ps1 -SkipInstall

# Только запуск без git
.\.venv\Scripts\python.exe app.py
```

**Требования:**
- Python 3.11
- Ollama запущен локально (`http://localhost:11434`)
- Нужные модели загружены: `ollama pull qwen2.5:14b-instruct-q4_K_M` и т.д.
- Файлы `assets/activate.wav`, `assets/deactivate.wav`

---

## Параллельное выполнение задач (llama-server)

Для параллельного выполнения независимых задач пайплайна используется `llama-server.exe` из llama.cpp.

### Установка llama-server

1. Скачать архивы:
   - `llama-b8946-bin-win-cuda-12.4-x64.zip`
   - `cudart-llama-bin-win-cuda-12.4-x64.zip`
2. Распаковать **оба** в `C:\llama-server\`
3. Убедиться что `C:\llama-server\llama-server.exe` существует

### Переменные окружения (опционально)

Если пути отличаются от дефолтных — задать перед запуском:

```powershell
$env:LLAMA_SERVER_EXE   = "C:\llama-server\llama-server.exe"
$env:LLAMA_SERVER_MODEL = "C:\Users\Genn_\.ollama\models\blobs\sha256-eabc98a9bcbfce7fd70f3e07de599f8fda98120fefed5881934161ede8bd1a41"
```

### Как работает

```
Executor.run(tasks)
  ├── independent tasks (depends_on=[])  →  asyncio.gather  →  llama-server :8080  (параллельно)
  └── dependent tasks   (depends_on=[…]) →  serial          →  Ollama        :11434
```

Сервер запускается автоматически перед параллельной волной и убивается сразу после — VRAM освобождается.

---

## Пути и данные

Репозиторий клонируется в `C:\Jarvis\`. Все рабочие файлы:

```
C:\Jarvis\
  data\memory.json       # долгосрочная память (в .gitignore)
  logs\router.jsonl      # лог всех роутов (в .gitignore)
  _tts_chunks\           # временные mp3-чанки TTS (создаётся автоматически)
  assets\                # звуки и референсный wav

C:\llama-server\         # llama-server.exe + CUDA DLL
C:\Users\Genn_\.ollama\models\blobs\  # GGUF-блобы моделей
```

---

## Заметки для разработки

- **Единый источник настроек:** все константы в `core/config.py`, все пути в `core/paths.py`
- **Ollama SDK:** `resp.message.content` (не `.get("message")`)
- **Память в чате:** `chat.py` вставляет `get_memory_context()` системным сообщением в каждый turn — Jarvis помнит пользователя без явного `route=memory`
- **Таймеры:** `tools/timer.py` использует `threading.Timer`; при срабатывании вызывает `say()` напрямую; `cancel_all()` вызывается при shutdown в `assistant.py`
- **TTS прерывание:** если пользователь заговорил во время ответа — `interrupt-listener` в `tts.py` глушит TTS, дособирает фразу и возвращает audio в `assistant.py` как `pending_audio`
- **brain/router.py:** legacy-файл с `parse_router_response()`, реальный роутинг делает `brain/ask.py → _route()`
- **llama-server backend:** `brain/client.py` содержит `set_backend("llama" | "ollama")` — Executor переключается автоматически; модель фиксирована при запуске сервера, параметр `model` игнорируется
