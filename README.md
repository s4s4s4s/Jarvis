# Jarvis

Локальный голосовой ассистент на Python 3.11 (Windows 11, RTX 4090).

## Стек
- STT: Whisper (локально)
- VAD: Silero VAD
- TTS: Edge TTS + pygame mixer (streaming по предложениям)
- LLM: Ollama локально
  - Router: qwen2.5:14b-instruct-q4_K_M
  - Fast: llama3.1:8b-instruct-q5_K_M
  - Deep: qwen2.5:32b-instruct-q4_K_M
- GUI: PySide6

## Структура
- `app.py` — точка входа
- `core/` — state, config, paths
- `voice/` — assistant, audio_core, stt, tts, wake, turn, sounds
- `brain/` — client, router, ask, history, prompts, agents/{chat,deep,web_agent,memory_agent}
- `tools/` — web_search, memory
- `ui/` — main_window, bridge
- `assets/` — звуки активации
- `logs/` — router.jsonl (в .gitignore)
- `data/` — memory.json (в .gitignore)

## Запуск
```powershell
python app.py
```
