import re
import time
import threading
from typing import Callable, Optional

from .audio_core import AudioCore
from .wake import WakeDetector
from .turn import TurnManager
from .stt import transcribe
from .tts import speak_and_handle, say, stop_speaking
from .sounds import play_activate, play_deactivate

from core.state import AssistantState
from core.config import (
    WAKE_PHRASES,
    CHUNK_SIZE,
    IDLE_TIMEOUT_SEC,
    POST_TTS_GRACE_SEC,
    IGNORE_PHRASES,
)
from brain.ask import ask_llm

FAREWELL_PATTERNS = [
    r"\bпока\b",
    r"\bпока[-\s]*пока\b",
    r"\bдо свидания\b",
    r"\bдо встречи\b",
    r"\bвыключись\b",
    r"\bотключись\b",
    r"\bстоп\b",
    r"\bхватит\b",
    r"\bспи\b",
    r"\bспокойной ночи\b",
    r"\bрежим ожидания\b",
    r"\bможешь спать\b",
    r"\bиди спать\b",
    r"\bотдыхай\b",
]

PASSIVE_COMMAND_PATTERNS = [
    r"\bперейди в пассивный режим\b",
    r"\bвключи пассивный режим\b",
    r"\bуйди в пассивный режим\b",
    r"\bперейди в режим ожидания\b",
    r"\bвключи режим ожидания\b",
    r"\bжди обращения\b",
    r"\bжди команду\b",
]

ACTIVE_COMMAND_PATTERNS = [
    r"\bперейди в активный режим\b",
    r"\bвключи активный режим\b",
    r"\bслушай постоянно\b",
    r"\bслушай без имени\b",
    r"\bне жди обращения\b",
    r"\bне жди имя\b",
]


def strip_wake_prefix(text: str) -> str:
    lower = text.lower().strip()
    for phrase in sorted(WAKE_PHRASES, key=len, reverse=True):
        if lower.startswith(phrase):
            return text[len(phrase):].strip(" ,.!?-")
    return text


def is_farewell(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(pattern, t) for pattern in FAREWELL_PATTERNS)


def matches_any_pattern(text: str, patterns: list[str]) -> bool:
    t = text.lower().strip()
    return any(re.search(pattern, t) for pattern in patterns)


def is_ignored_phrase(text: str) -> bool:
    t = text.lower().strip()
    return any(phrase in t for phrase in IGNORE_PHRASES)


def _safe_call(callback, *args, **kwargs):
    if callback is None:
        return
    try:
        callback(*args, **kwargs)
    except Exception as e:
        print(f"[assistant] callback error: {e}")


def _emit_status(
    state: AssistantState,
    on_state: Optional[Callable[[AssistantState], None]] = None,
    on_system_log: Optional[Callable[[str], None]] = None,
):
    _safe_call(on_state, state)
    if state == AssistantState.IDLE:
        _safe_call(on_system_log, "[state] IDLE\n")
    elif state == AssistantState.LISTENING:
        _safe_call(on_system_log, "[state] LISTENING\n")
    elif state == AssistantState.THINKING:
        _safe_call(on_system_log, "[state] THINKING\n")
    elif state == AssistantState.SPEAKING:
        _safe_call(on_system_log, "[state] SPEAKING\n")
    elif state == AssistantState.INTERRUPT_LISTEN:
        _safe_call(on_system_log, "[state] INTERRUPT_LISTEN\n")


def main(
    stop_event: Optional[threading.Event] = None,
    on_state: Optional[Callable[[AssistantState], None]] = None,
    on_user_text: Optional[Callable[[str], None]] = None,
    on_assistant_text: Optional[Callable[[str], None]] = None,
    on_system_log: Optional[Callable[[str], None]] = None,
):
    stop_event = stop_event or threading.Event()
    pending_audio = None
    post_tts_grace_until = 0.0

    audio_core = AudioCore()
    audio_core.start()
    wake_detector = WakeDetector()
    turn_manager = TurnManager(CHUNK_SIZE)
    chunk_iter = audio_core.stream_chunks(stop_event=stop_event)

    _safe_call(on_system_log, "\n=== Jarvis запущен ===\n")
    active_mode = True
    play_activate()
    _emit_status(AssistantState.LISTENING, on_state, on_system_log)
    _safe_call(on_system_log, "🎙️ Активный режим. Слушаю...\n")

    try:
        while not stop_event.is_set():
            if not active_mode and pending_audio is None:
                _emit_status(AssistantState.IDLE, on_state, on_system_log)
                _safe_call(on_system_log, "💤 Жду обращения...\n")
                while not stop_event.is_set():
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        return
                    if wake_detector.process_chunk(chunk):
                        play_activate()
                        active_mode = True
                        _emit_status(AssistantState.LISTENING, on_state, on_system_log)
                        _safe_call(on_system_log, "🎙️ Активный режим. Слушаю...\n")
                        pre_roll = audio_core.get_pre_roll()
                        pending_audio = turn_manager.collect_utterance(
                            chunk_iter,
                            initial_frames=pre_roll,
                            stop_event=stop_event,
                        )
                        break

            if stop_event.is_set():
                break

            if active_mode:
                if pending_audio is not None:
                    audio = pending_audio
                    pending_audio = None
                else:
                    now = time.time()
                    if now < post_tts_grace_until:
                        time.sleep(0.05)
                        continue
                    _emit_status(AssistantState.LISTENING, on_state, on_system_log)
                    audio = turn_manager.collect_with_timeout(
                        chunk_iter,
                        idle_timeout_sec=IDLE_TIMEOUT_SEC,
                        stop_event=stop_event,
                    )

                if stop_event.is_set():
                    break

                if audio is None:
                    active_mode = False
                    play_deactivate()
                    _emit_status(AssistantState.IDLE, on_state, on_system_log)
                    _safe_call(on_system_log, "💤 Тишина, возвращаюсь в ожидание...\n")
                    continue

                _emit_status(AssistantState.THINKING, on_state, on_system_log)
                text = transcribe(audio)

                if stop_event.is_set():
                    break
                if not text or len(text) < 2:
                    continue

                text = strip_wake_prefix(text)
                if not text:
                    continue

                if is_ignored_phrase(text):
                    _safe_call(on_system_log, f"[игнорирую фон] {text}\n")
                    continue

                _safe_call(on_user_text, text)

                if matches_any_pattern(text, ACTIVE_COMMAND_PATTERNS):
                    _safe_call(on_system_log, "🟢 Активный режим уже включён\n")
                    post_tts_grace_until = time.time() + POST_TTS_GRACE_SEC
                    continue

                if matches_any_pattern(text, PASSIVE_COMMAND_PATTERNS) or is_farewell(text):
                    active_mode = False
                    play_deactivate()
                    _emit_status(AssistantState.IDLE, on_state, on_system_log)
                    _safe_call(on_system_log, "💤 Пассивный режим включён\n")
                    _safe_call(on_system_log, "💤 Жду обращения...\n")
                    continue

                # --- LLM с параллельным filler'ом ---
                result = ask_llm(text)
                if stop_event.is_set():
                    break

                if result.filler:
                    _safe_call(on_system_log, f"[filler] {result.filler}\n")
                    _safe_call(on_assistant_text, result.filler)
                    _emit_status(AssistantState.SPEAKING, on_state, on_system_log)
                    say(result.filler, stop_event=stop_event)

                if stop_event.is_set():
                    break

                answer = result.get_answer(timeout=120.0)
                if stop_event.is_set():
                    break
                if not answer:
                    continue

                _safe_call(on_assistant_text, answer)
                _emit_status(AssistantState.INTERRUPT_LISTEN, on_state, on_system_log)
                _safe_call(on_system_log, "🟣 Слушаю перебивание во время TTS...\n")

                interrupted = speak_and_handle(
                    answer,
                    stop_event=stop_event,
                    audio_core=audio_core,
                )

                if stop_event.is_set():
                    break

                if interrupted is not None:
                    # Пользователь перебил — сразу обрабатываем его аудио
                    _safe_call(on_system_log, "[assistant] TTS прерван — обрабатываю реплику...\n")
                    _emit_status(AssistantState.THINKING, on_state, on_system_log)
                    pending_audio = interrupted
                else:
                    # TTS доиграл штатно
                    _emit_status(AssistantState.SPEAKING, on_state, on_system_log)
                    _safe_call(on_system_log, "Говорю...\n")
                    post_tts_grace_until = time.time() + POST_TTS_GRACE_SEC

    except KeyboardInterrupt:
        _safe_call(on_system_log, "\nJarvis выключен.\n")
    except Exception as e:
        _safe_call(on_system_log, f"[assistant] ERROR: {e}\n")
        raise
    finally:
        stop_event.set()
        audio_core.request_stop()
        stop_speaking()
        _emit_status(AssistantState.IDLE, on_state, on_system_log)
        _safe_call(on_system_log, "Jarvis остановлен.\n")
