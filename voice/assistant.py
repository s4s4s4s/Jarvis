# voice/assistant.py
"""
Главный голосовой цикл Jarvis.
Запускается из ui/bridge.py через main(stop_event, callbacks...).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from core.config import CHUNK_SIZE, POST_TTS_GRACE_SEC, POST_INTERRUPT_GRACE_SEC, IDLE_TIMEOUT_SEC
from voice.audio_core import AudioCore
from voice.wake import WakeDetector
from voice.turn import TurnManager
from voice.stt import transcribe
from voice import tts
from brain.ask import ask_llm

try:
    from voice.state import AssistantState
except Exception:
    class AssistantState:  # type: ignore
        IDLE = "IDLE"
        LISTENING = "LISTENING"
        THINKING = "THINKING"
        SPEAKING = "SPEAKING"
        INTERRUPT_LISTEN = "INTERRUPT_LISTEN"


def main(
    stop_event: Optional[threading.Event] = None,
    on_state: Optional[Callable] = None,
    on_user_text: Optional[Callable[[str], None]] = None,
    on_assistant_text: Optional[Callable[[str], None]] = None,
    on_system_log: Optional[Callable[[str], None]] = None,
) -> None:
    if stop_event is None:
        stop_event = threading.Event()

    def _log(msg: str) -> None:
        print(msg)
        if on_system_log:
            try:
                on_system_log(msg)
            except Exception:
                pass

    def _state(s) -> None:
        if on_state:
            try:
                on_state(s)
            except Exception:
                pass

    # ── Ollama health-check ──────────────────────────────────────────────────
    from brain.client import is_ollama_available
    if not is_ollama_available():
        msg = "Сэр, Ollama недоступна. Убедитесь, что сервис запущен на порту 11434."
        _log(f"[assistant] {msg}")
        tts.say(msg, stop_event=stop_event)
        return

    _log("[assistant] Ollama доступна. Запускаю голосовой цикл.")

    audio_core = AudioCore()
    wake_detector = WakeDetector()
    turn_manager = TurnManager(chunk_size=CHUNK_SIZE)

    audio_core.start()
    _log("[assistant] AudioCore запущен.")
    _state(AssistantState.IDLE)

    def _chunk_iter(tap_q):
        from queue import Empty
        while not stop_event.is_set():
            try:
                chunk = tap_q.get(timeout=0.2)
                yield chunk
            except Empty:
                continue

    try:
        while not stop_event.is_set():
            # ── Фаза 1: Ожидание wake-word ───────────────────────────────────
            _state(AssistantState.IDLE)
            wake_tap = audio_core.create_tap()
            _log("[assistant] Жду wake-word...")
            woke = False
            for chunk in _chunk_iter(wake_tap):
                if wake_detector.process_chunk(chunk):
                    woke = True
                    break
            audio_core.remove_tap(wake_tap)
            if not woke or stop_event.is_set():
                break

            # ── Фаза 2: Запись utterance ─────────────────────────────────────
            _log("[assistant] Wake! Слушаю команду...")
            _state(AssistantState.LISTENING)
            listen_tap = audio_core.create_tap()
            audio = turn_manager.collect_with_timeout(
                _chunk_iter(listen_tap),
                idle_timeout_sec=IDLE_TIMEOUT_SEC,
                stop_event=stop_event,
            )
            audio_core.remove_tap(listen_tap)

            if audio is None or stop_event.is_set():
                _log("[assistant] Utterance не получен, возврат в idle.")
                continue

            # ── Фаза 3: STT ──────────────────────────────────────────────────
            text = transcribe(audio)
            if not text or not text.strip():
                _log("[assistant] STT вернул пустую строку.")
                continue

            _log(f"[assistant] STT: {text}")
            if on_user_text:
                try:
                    on_user_text(text)
                except Exception:
                    pass

            # ── Фаза 4: LLM (async) + filler ────────────────────────────────
            _state(AssistantState.THINKING)
            ask_result = ask_llm(text)

            if ask_result.filler:
                _state(AssistantState.SPEAKING)
                tts.say(ask_result.filler, stop_event=stop_event)

            # ── Фаза 5: Дожидаемся ответа LLM ───────────────────────────────
            _state(AssistantState.THINKING)
            answer = ask_result.get_answer(timeout=120.0)
            if not answer:
                _log("[assistant] LLM вернул пустой ответ.")
                continue

            _log(f"[assistant] LLM: {answer[:120]}")
            if on_assistant_text:
                try:
                    on_assistant_text(answer)
                except Exception:
                    pass

            # ── Фаза 6: TTS с возможностью перебить ─────────────────────────
            _state(AssistantState.SPEAKING)
            interrupted_audio = tts.speak_and_handle(
                answer,
                stop_event=stop_event,
                audio_core=audio_core,
            )

            if stop_event.is_set():
                break

            if interrupted_audio is not None:
                # Пользователь перебил — сразу обрабатываем его фразу
                _log("[assistant] Пользователь перебил TTS.")
                _state(AssistantState.INTERRUPT_LISTEN)
                int_text = transcribe(interrupted_audio)
                if int_text and int_text.strip():
                    _log(f"[assistant] Перебивание STT: {int_text}")
                    if on_user_text:
                        try:
                            on_user_text(int_text)
                        except Exception:
                            pass
                    _state(AssistantState.THINKING)
                    int_result = ask_llm(int_text)
                    if int_result.filler:
                        _state(AssistantState.SPEAKING)
                        tts.say(int_result.filler, stop_event=stop_event)
                    _state(AssistantState.THINKING)
                    int_answer = int_result.get_answer(timeout=120.0)
                    if int_answer:
                        if on_assistant_text:
                            try:
                                on_assistant_text(int_answer)
                            except Exception:
                                pass
                        _state(AssistantState.SPEAKING)
                        tts.speak_and_handle(
                            int_answer,
                            stop_event=stop_event,
                            audio_core=audio_core,
                        )
            else:
                import time
                time.sleep(POST_TTS_GRACE_SEC)

    finally:
        audio_core.stop()
        _log("[assistant] AudioCore остановлен. Выход.")
        _state(AssistantState.IDLE)
