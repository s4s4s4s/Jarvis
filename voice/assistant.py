from __future__ import annotations

import threading
from typing import Callable, Optional

from core.config import CHUNK_SIZE, POST_TTS_GRACE_SEC, IDLE_TIMEOUT_SEC
from voice.audio_core import AudioCore
from voice.wake import WakeDetector
from voice.turn import TurnManager
from voice import tts
from voice.state import AssistantState
from brain.ask import ask_llm

_running_lock = threading.Lock()
_is_running = False


def main(
    stop_event: Optional[threading.Event] = None,
    on_state: Optional[Callable] = None,
    on_user_text: Optional[Callable[[str], None]] = None,
    on_assistant_text: Optional[Callable[[str], None]] = None,
    on_system_log: Optional[Callable[[str], None]] = None,
) -> None:
    global _is_running
    with _running_lock:
        if _is_running:
            print("[assistant] Уже запущен, игнорирую повторный вызов.")
            return
        _is_running = True

    if stop_event is None:
        stop_event = threading.Event()

    def _log(msg: str) -> None:
        print(msg)

    def _state(s) -> None:
        if on_state:
            try:
                on_state(s)
            except Exception:
                pass

    audio_core = AudioCore()

    try:
        from brain.client import is_ollama_available
        if not is_ollama_available():
            msg = "Сэр, Ollama недоступна. Убедитесь, что сервис запущен на порту 11434."
            _log(f"[assistant] {msg}")
            tts.say(msg, stop_event=stop_event)
            return

        _log("[assistant] Ollama доступна.")

        # ── Инициализация: загружаем всё ДО старта любых циклов ─────────
        _log("[assistant] Инициализация моделей...")
        import numpy as np
        from voice.stt import vad_prob
        vad_prob(np.zeros(512, dtype="float32"))  # триггерит _ensure_models()
        if stop_event.is_set():
            return
        _log("[assistant] Модели готовы.")

        # FIX: подключаем TTS-callback для таймеров
        # Используем замыкание на stop_event чтобы таймер не говорил
        # когда ассистент уже остановлен
        from tools.timer import set_fire_callback
        set_fire_callback(lambda msg: tts.say(msg, stop_event=stop_event))
        _log("[assistant] Timer TTS callback установлен.")

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

        while not stop_event.is_set():
            _state(AssistantState.IDLE)
            wake_tap = audio_core.create_tap(pre_roll=False)
            _log("[assistant] Жду wake-word...")
            woke = False
            for chunk in _chunk_iter(wake_tap):
                if wake_detector.process_chunk(chunk):
                    woke = True
                    break
            audio_core.remove_tap(wake_tap)
            if not woke or stop_event.is_set():
                break

            _log("[assistant] Wake! Слушаю команду...")
            _state(AssistantState.LISTENING)
            listen_tap = audio_core.create_tap(pre_roll=True)
            audio = turn_manager.collect_with_timeout(
                _chunk_iter(listen_tap),
                idle_timeout_sec=IDLE_TIMEOUT_SEC,
                stop_event=stop_event,
            )
            audio_core.remove_tap(listen_tap)

            if audio is None or stop_event.is_set():
                _log("[assistant] Utterance не получен, возврат в idle.")
                continue

            from voice.stt import transcribe
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

            _state(AssistantState.THINKING)
            ask_result = ask_llm(text)

            if ask_result.filler:
                _state(AssistantState.SPEAKING)
                tts.say(ask_result.filler, stop_event=stop_event)

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

            _state(AssistantState.SPEAKING)
            interrupted_audio = tts.speak_and_handle(
                answer,
                stop_event=stop_event,
                audio_core=audio_core,
            )

            if stop_event.is_set():
                break

            if interrupted_audio is not None:
                _log("[assistant] Пользователь перебил TTS.")
                _state(AssistantState.INTERRUPT_LISTEN)
                from voice.stt import transcribe
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
        with _running_lock:
            _is_running = False
        audio_core.stop()
        # FIX: отменяем все активные таймеры при завершении ассистента,
        # чтобы daemon-потоки не пытались вызвать tts.say() после остановки
        try:
            from tools.timer import cancel_all
            cancelled = cancel_all()
            if cancelled:
                print(f"[assistant] Отменено таймеров при выходе: {cancelled}")
        except Exception:
            pass
        print("[assistant] AudioCore остановлен. Выход.")
        _state(AssistantState.IDLE)
