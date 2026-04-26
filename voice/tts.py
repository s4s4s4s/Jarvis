"""
voice/tts.py — Edge TTS с параллельным streaming и перебиванием голосом через единый AudioCore.
"""
import asyncio
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import numpy as np
import edge_tts
import pygame

from core.config import (
    EDGE_VOICE,
    EDGE_RATE,
    SAMPLE_RATE_MIC,
    CHUNK_SIZE,
    TURN_VAD_TRIGGER,
)
from core.paths import TTS_CHUNKS

_CHUNK_ROOT = str(TTS_CHUNKS)
_STOP_SENTINEL = object()
_stop_flag = False
_playing = False
_mixer_ready = False
_state_lock = threading.Lock()

_LISTENER_WAIT_TIMEOUT = 15.0


def _set_playing(value: bool) -> None:
    global _playing
    with _state_lock:
        _playing = value


def _get_stop_flag() -> bool:
    with _state_lock:
        return _stop_flag


def _set_stop_flag(value: bool) -> None:
    global _stop_flag
    with _state_lock:
        _stop_flag = value


def _should_stop(stop_event: Optional[threading.Event] = None) -> bool:
    if _get_stop_flag():
        return True
    if stop_event is not None and stop_event.is_set():
        return True
    return False


def _ensure_chunk_root():
    Path(_CHUNK_ROOT).mkdir(parents=True, exist_ok=True)


def _new_session_dir() -> str:
    _ensure_chunk_root()
    session = os.path.join(_CHUNK_ROOT, uuid.uuid4().hex[:8])
    Path(session).mkdir(parents=True, exist_ok=True)
    return session


def _cleanup_session_dir(session_dir: str) -> None:
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as e:
        print(f"[TTS] Ошибка очистки session {session_dir}: {e}")


def _cleanup_orphan_sessions(keep: Optional[str] = None) -> None:
    try:
        if not os.path.exists(_CHUNK_ROOT):
            return
        for name in os.listdir(_CHUNK_ROOT):
            full = os.path.join(_CHUNK_ROOT, name)
            if keep and os.path.abspath(full) == os.path.abspath(keep):
                continue
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            elif name.endswith(".mp3"):
                try:
                    os.remove(full)
                except Exception:
                    pass
    except Exception as e:
        print(f"[TTS] Ошибка очистки orphan sessions: {e}")


def _ensure_mixer():
    global _mixer_ready
    if _mixer_ready and pygame.mixer.get_init():
        return
    try:
        if not pygame.get_init():
            pygame.init()
    except Exception:
        pass
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        _mixer_ready = True
    except Exception as e:
        _mixer_ready = False
        print(f"[TTS] Не удалось инициализировать pygame.mixer: {e}")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part[-1] not in ".!?":
            part += "."
        sentences.append(part)
    if not sentences and text:
        return [text if text[-1] in ".!?" else text + "."]
    return sentences


async def _save_chunk_with_retry(sentence: str, chunk_path: str, retries: int = 3) -> bool:
    last_err = None
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(sentence, EDGE_VOICE, rate=EDGE_RATE)
            await communicate.save(chunk_path)
            return True
        except PermissionError as e:
            last_err = e
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception:
                pass
            await asyncio.sleep(0.15 * (attempt + 1))
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.1)
    print(f"[TTS] Не удалось сохранить чанк {chunk_path}: {last_err}")
    return False


async def _generate_chunks(
    text: str,
    session_dir: str,
    chunk_queue: Queue,
    stop_event: Optional[threading.Event] = None,
):
    try:
        sentences = _split_sentences(text)
        print(f"[TTS] Генерирую {len(sentences)} чанков")
        for idx, sentence in enumerate(sentences):
            if _should_stop(stop_event):
                print("[TTS] Генерация прервана")
                break
            chunk_path = os.path.join(session_dir, f"chunk_{idx}.mp3")
            ok = await _save_chunk_with_retry(sentence, chunk_path, retries=3)
            if _should_stop(stop_event):
                print("[TTS] stop после генерации чанка")
                break
            if not ok:
                continue
            chunk_queue.put(chunk_path)
            print(f"[TTS] Чанк {idx} готов: {sentence[:60]}...")
        chunk_queue.put(_STOP_SENTINEL)
        print("[TTS] Генерация всех чанков завершена")
    except Exception as e:
        print(f"[TTS] Ошибка генерации чанков: {e}")
        chunk_queue.put(_STOP_SENTINEL)


def _gen_thread_target(text: str, session_dir: str, chunk_queue: Queue,
                       stop_event: Optional[threading.Event]):
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _generate_chunks(text, session_dir, chunk_queue, stop_event=stop_event)
        )
    finally:
        loop.close()


def _release_current_chunk():
    try:
        if pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
    except Exception:
        pass


def _playback_worker(chunk_queue: Queue, stop_event: Optional[threading.Event] = None):
    chunk_idx = 0
    _set_playing(False)
    while not _should_stop(stop_event):
        try:
            item = chunk_queue.get(timeout=0.20)
        except Empty:
            continue
        if item is _STOP_SENTINEL:
            print("[TTS] Очередь playback завершена")
            break
        chunk_path = item
        if _should_stop(stop_event):
            break
        if not os.path.exists(chunk_path):
            print(f"[TTS] Чанк не найден: {chunk_path}")
            continue
        try:
            _ensure_mixer()
            if not pygame.mixer.get_init():
                print("[TTS] mixer не инициализирован, playback пропущен")
                continue
            print(f"[TTS] Играю чанк {chunk_idx}: {os.path.getsize(chunk_path)} байт")
            pygame.mixer.music.load(chunk_path)
            pygame.mixer.music.play()
            _set_playing(True)
            while pygame.mixer.music.get_busy():
                if _should_stop(stop_event):
                    print("[TTS] Воспроизведение прервано")
                    _release_current_chunk()
                    break
                time.sleep(0.05)
            _release_current_chunk()
            _set_playing(False)
            chunk_idx += 1
        except Exception as e:
            _set_playing(False)
            _release_current_chunk()
            print(f"[TTS] Ошибка воспроизведения чанка {chunk_idx}: {e}")
    _release_current_chunk()
    _set_playing(False)


def _interrupt_listener_from_audio_core(
    audio_core,
    stop_event: Optional[threading.Event],
    interrupt_event: threading.Event,
    interrupted_audio_box: dict,
):
    from voice.turn import TurnManager
    from voice.audio_core import CHUNK_SIZE as _CS

    tap_q = audio_core.create_tap()

    def _chunk_iter():
        while not interrupt_event.is_set() and not (stop_event and stop_event.is_set()):
            try:
                item = tap_q.get(timeout=0.20)
                if item is _STOP_SENTINEL:
                    return
                yield item
            except Exception:
                return

    try:
        tm = TurnManager(chunk_size=_CS)
        first_chunk_seen = threading.Event()

        def _patched_iter():
            for chunk in _chunk_iter():
                prob_val = None
                try:
                    from voice.stt import vad_prob
                    prob_val = vad_prob(chunk)
                except Exception:
                    pass
                if prob_val is not None and prob_val >= TURN_VAD_TRIGGER and not first_chunk_seen.is_set():
                    first_chunk_seen.set()
                    _set_stop_flag(True)
                    _release_current_chunk()
                    print("[TTS] Речь обнаружена — TTS остановлен немедленно, досбираю фразу...")
                yield chunk

        audio = tm.collect_utterance(_patched_iter(), stop_event=stop_event)
        if audio is not None and len(audio) > 0:
            interrupted_audio_box["audio"] = audio
            interrupt_event.set()
            print("[TTS] Перебивание зафиксировано, аудио передано")
    finally:
        audio_core.remove_tap(tap_q)


def _run_streaming(
    text: str,
    stop_event: Optional[threading.Event] = None,
    allow_interrupt: bool = False,
    audio_core=None,
):
    if not text or not text.strip():
        return None

    _set_stop_flag(False)
    _set_playing(False)

    session_dir = _new_session_dir()
    _cleanup_orphan_sessions(keep=session_dir)
    _ensure_mixer()
    print(f"[TTS] Начинаю streaming: {text[:80]}...")
    chunk_queue = Queue()
    interrupt_event = threading.Event()
    interrupted_audio_box = {"audio": None}

    gen_thread = threading.Thread(
        target=_gen_thread_target,
        args=(text, session_dir, chunk_queue, stop_event),
        daemon=True,
    )
    play_thread = threading.Thread(
        target=lambda: _playback_worker(chunk_queue, stop_event=stop_event),
        daemon=True,
    )
    listener_thread = None
    if allow_interrupt and audio_core is not None:
        listener_thread = threading.Thread(
            target=lambda: _interrupt_listener_from_audio_core(
                audio_core=audio_core,
                stop_event=stop_event,
                interrupt_event=interrupt_event,
                interrupted_audio_box=interrupted_audio_box,
            ),
            daemon=True,
        )
    gen_thread.start()
    play_thread.start()
    if listener_thread is not None:
        listener_thread.start()
    try:
        while True:
            tts_done = not gen_thread.is_alive() and not play_thread.is_alive()
            listener_done = listener_thread is None or not listener_thread.is_alive()

            if _get_stop_flag() and listener_thread is not None:
                try:
                    chunk_queue.put_nowait(_STOP_SENTINEL)
                except Exception:
                    pass
                gen_thread.join(timeout=0.5)
                play_thread.join(timeout=0.5)
                _release_current_chunk()
                _set_playing(False)
                if not listener_done:
                    print("[TTS] Жду listener — досбирает фразу...")
                    listener_thread.join(timeout=_LISTENER_WAIT_TIMEOUT)
                    if listener_thread.is_alive():
                        print("[TTS] listener timeout — аудио не получено")
                print("[TTS] Streaming завершён (прервано)")
                return interrupted_audio_box["audio"]

            if stop_event is not None and stop_event.is_set():
                _set_stop_flag(True)
                try:
                    chunk_queue.put_nowait(_STOP_SENTINEL)
                except Exception:
                    pass
                _release_current_chunk()
                gen_thread.join(timeout=0.10)
                play_thread.join(timeout=0.10)
                if listener_thread is not None:
                    listener_thread.join(timeout=0.10)
                _set_playing(False)
                print("[TTS] Streaming остановлен внешним stop_event")
                return None

            if tts_done and listener_done:
                _set_playing(False)
                _release_current_chunk()
                print("[TTS] Streaming завершён штатно")
                return interrupted_audio_box["audio"]

            time.sleep(0.05)
    finally:
        _cleanup_session_dir(session_dir)


def say(text: str, stop_event: Optional[threading.Event] = None):
    _run_streaming(text, stop_event=stop_event, allow_interrupt=False, audio_core=None)


def speak_and_handle(
    text: str,
    stop_event: Optional[threading.Event] = None,
    audio_core=None,
):
    return _run_streaming(
        text,
        stop_event=stop_event,
        allow_interrupt=True,
        audio_core=audio_core,
    )


def stop_speaking():
    _set_stop_flag(True)
    _release_current_chunk()
    print("[TTS] Остановлен внешним вызовом")


def is_speaking() -> bool:
    with _state_lock:
        return _playing
