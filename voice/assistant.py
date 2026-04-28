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


def _speak_turn(
    ask_result,
    stop_event,
    audio_core,
    on_assistant_text: Optional[Callable[[str], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    set_state: Optional[Callable] = None,
) -> Optional[object]:
    """
    Shared helper: send full answer to chat UI, short voice_reply to TTS.
    Uses ask_result._timeout so long-running routes (test/plan) never hit
    the old hardcoded 120 s limit.
    Returns interrupted_audio (or None).
    """
    def _log(msg): log_fn and log_fn(msg)
    def _state(s): set_state and set_state(s)

    # Use per-route timeout stored in AskResult (set by ask_llm)
    answer = ask_result.get_answer()  # timeout comes from ask_result._timeout
    if not answer:
        _log("[assistant] LLM \u0432\u0435\u0440\u043d\u0443\u043b \u043f\u0443\u0441\u0442\u043e\u0439 \u043e\u0442\u0432\u0435\u0442.")
        return None

    voice_reply = ask_result.get_voice_reply()

    _log(f"[assistant] Chat: {answer[:120]}{'...' if len(answer) > 120 else ''}")
    if on_assistant_text:
        try:
            on_assistant_text(answer)
        except Exception:
            pass

    _log(f"[assistant] TTS: {voice_reply}")
    _state(AssistantState.SPEAKING)
    return tts.speak_and_handle(
        voice_reply,
        stop_event=stop_event,
        audio_core=audio_core,
    )


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
            print("[assistant] \u0423\u0436\u0435 \u0437\u0430\u043f\u0443\u0449\u0435\u043d, \u0438\u0433\u043d\u043e\u0440\u0438\u0440\u0443\u044e \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u044b\u0439 \u0432\u044b\u0437\u043e\u0432.")
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
            msg = "\u0421\u044d\u0440, Ollama \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430. \u0423\u0431\u0435\u0434\u0438\u0442\u0435\u0441\u044c, \u0447\u0442\u043e \u0441\u0435\u0440\u0432\u0438\u0441 \u0437\u0430\u043f\u0443\u0449\u0435\u043d \u043d\u0430 \u043f\u043e\u0440\u0442\u0443 11434."
            _log(f"[assistant] {msg}")
            tts.say(msg, stop_event=stop_event)
            return

        _log("[assistant] Ollama \u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430.")

        _log("[assistant] \u0418\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0435\u0439...")
        import numpy as np
        from voice.stt import vad_prob
        vad_prob(np.zeros(512, dtype="float32"))
        if stop_event.is_set():
            return
        _log("[assistant] \u041c\u043e\u0434\u0435\u043b\u0438 \u0433\u043e\u0442\u043e\u0432\u044b.")

        from tools.timer import set_fire_callback
        set_fire_callback(lambda msg: tts.say(msg, stop_event=stop_event))
        _log("[assistant] Timer TTS callback \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d.")

        wake_detector = WakeDetector()
        turn_manager = TurnManager(chunk_size=CHUNK_SIZE)

        audio_core.start()
        _log("[assistant] AudioCore \u0437\u0430\u043f\u0443\u0449\u0435\u043d.")
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
            _log("[assistant] \u0416\u0434\u0443 wake-word...")
            woke = False
            for chunk in _chunk_iter(wake_tap):
                if wake_detector.process_chunk(chunk):
                    woke = True
                    break
            audio_core.remove_tap(wake_tap)
            if not woke or stop_event.is_set():
                break

            _log("[assistant] Wake! \u0421\u043b\u0443\u0448\u0430\u044e \u043a\u043e\u043c\u0430\u043d\u0434\u0443...")
            _state(AssistantState.LISTENING)
            listen_tap = audio_core.create_tap(pre_roll=True)
            audio = turn_manager.collect_with_timeout(
                _chunk_iter(listen_tap),
                idle_timeout_sec=IDLE_TIMEOUT_SEC,
                stop_event=stop_event,
            )
            audio_core.remove_tap(listen_tap)

            if audio is None or stop_event.is_set():
                _log("[assistant] Utterance \u043d\u0435 \u043f\u043e\u043b\u0443\u0447\u0435\u043d, \u0432\u043e\u0437\u0432\u0440\u0430\u0442 \u0432 idle.")
                continue

            from voice.stt import transcribe
            text = transcribe(audio)
            if not text or not text.strip():
                _log("[assistant] STT \u0432\u0435\u0440\u043d\u0443\u043b \u043f\u0443\u0441\u0442\u0443\u044e \u0441\u0442\u0440\u043e\u043a\u0443.")
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
                _log(f"[assistant] Filler: {ask_result.filler}")
                if on_assistant_text:
                    try:
                        on_assistant_text(ask_result.filler)
                    except Exception:
                        pass
                _state(AssistantState.SPEAKING)
                tts.say(ask_result.filler, stop_event=stop_event)

            _state(AssistantState.THINKING)

            interrupted_audio = _speak_turn(
                ask_result,
                stop_event=stop_event,
                audio_core=audio_core,
                on_assistant_text=on_assistant_text,
                log_fn=_log,
                set_state=_state,
            )

            if stop_event.is_set():
                break

            if interrupted_audio is not None:
                _log("[assistant] \u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c \u043f\u0435\u0440\u0435\u0431\u0438\u043b TTS.")
                _state(AssistantState.INTERRUPT_LISTEN)
                from voice.stt import transcribe
                int_text = transcribe(interrupted_audio)
                if int_text and int_text.strip():
                    _log(f"[assistant] \u041f\u0435\u0440\u0435\u0431\u0438\u0432\u0430\u043d\u0438\u0435 STT: {int_text}")
                    if on_user_text:
                        try:
                            on_user_text(int_text)
                        except Exception:
                            pass
                    _state(AssistantState.THINKING)
                    int_result = ask_llm(int_text)
                    if int_result.filler:
                        _log(f"[assistant] Filler: {int_result.filler}")
                        if on_assistant_text:
                            try:
                                on_assistant_text(int_result.filler)
                            except Exception:
                                pass
                        _state(AssistantState.SPEAKING)
                        tts.say(int_result.filler, stop_event=stop_event)
                    _state(AssistantState.THINKING)
                    _speak_turn(
                        int_result,
                        stop_event=stop_event,
                        audio_core=audio_core,
                        on_assistant_text=on_assistant_text,
                        log_fn=_log,
                        set_state=_state,
                    )
            else:
                import time
                time.sleep(POST_TTS_GRACE_SEC)

    finally:
        with _running_lock:
            _is_running = False
        audio_core.stop()
        try:
            from tools.timer import cancel_all
            cancelled = cancel_all()
            if cancelled:
                print(f"[assistant] \u041e\u0442\u043c\u0435\u043d\u0435\u043d\u043e \u0442\u0430\u0439\u043c\u0435\u0440\u043e\u0432 \u043f\u0440\u0438 \u0432\u044b\u0445\u043e\u0434\u0435: {cancelled}")
        except Exception:
            pass
        print("[assistant] AudioCore \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d. \u0412\u044b\u0445\u043e\u0434.")
        _state(AssistantState.IDLE)
