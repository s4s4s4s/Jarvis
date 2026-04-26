"""
voice/tts.py — Edge TTS: цельный текст одним запросом, потом воспроизводим один файл.
"""
import asyncio
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import edge_tts
import pygame

from core.config import EDGE_VOICE, EDGE_RATE, SAMPLE_RATE_MIC, TURN_VAD_TRIGGER, CHUNK_SIZE
from core.paths import TTS_CHUNKS
# FIX (audit 3): импортируем sentinel из audio_core, чтобы _listen_thread
# мог корректно сравнивать через `is`. Раньше в tts.py был свой _STOP_SENTINEL,
# который никогда не совпадал с тем, что кладёт audio_core.remove_tap()/stop().
from voice.audio_core import _STOP_SENTINEL as _AUDIO_STOP_SENTINEL

_CHUNK_ROOT = str(TTS_CHUNKS)
# Локальный sentinel — для внутренней очереди _gen_thread → _play_thread.
# Для tap-очереди от audio_core используем _AUDIO_STOP_SENTINEL.
_TTS_STOP_SENTINEL = object()
_stop_flag = False
_playing = False
_mixer_ready = False
_state_lock = threading.Lock()
_LISTENER_WAIT_TIMEOUT = 15.0


def _set_playing(v):  global _playing;   _playing = v
def _get_stop():
    with _state_lock: return _stop_flag
def _set_stop(v):
    global _stop_flag
    with _state_lock: _stop_flag = v
def _should_stop(ev=None):
    return _get_stop() or (ev is not None and ev.is_set())


def _ensure_chunk_root():
    Path(_CHUNK_ROOT).mkdir(parents=True, exist_ok=True)


def _new_session_dir():
    _ensure_chunk_root()
    p = os.path.join(_CHUNK_ROOT, uuid.uuid4().hex[:8])
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _cleanup(path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _cleanup_orphans(keep=None):
    """Удаляет старые сессионные директории. Вызывать только при старте _run."""
    try:
        if not os.path.exists(_CHUNK_ROOT):
            return
        for name in os.listdir(_CHUNK_ROOT):
            full = os.path.join(_CHUNK_ROOT, name)
            if keep and os.path.abspath(full) == os.path.abspath(keep):
                continue
            shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) else None
    except Exception:
        pass


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
        print(f"[TTS] mixer init failed: {e}")


def _release():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception:
        pass


async def _synthesize(text: str, path: str) -> bool:
    """Синтезирует ВЕСЬ text в один mp3-файл."""
    for attempt in range(3):
        try:
            comm = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE)
            await comm.save(path)
            return True
        except PermissionError:
            try:
                os.remove(path)
            except Exception:
                pass
            await asyncio.sleep(0.15 * (attempt + 1))
        except Exception as e:
            print(f"[TTS] synthesize attempt {attempt+1} failed: {e}")
            await asyncio.sleep(0.1)
    return False


def _gen_thread(text: str, session_dir: str, q: Queue, stop_ev):
    """Генерирует один mp3 и кладёт путь в очередь."""
    loop = asyncio.new_event_loop()
    try:
        if _should_stop(stop_ev):
            q.put(_STOP_SENTINEL)
            return
        mp3 = os.path.join(session_dir, "speech.mp3")
        ok = loop.run_until_complete(_synthesize(text, mp3))
        if ok and not _should_stop(stop_ev):
            q.put(mp3)
        q.put(_TTS_STOP_SENTINEL)
    finally:
        loop.close()


def _play_thread(q: Queue, stop_ev):
    _set_playing(False)
    while not _should_stop(stop_ev):
        try:
            item = q.get(timeout=0.2)
        except Empty:
            continue
        if item is _TTS_STOP_SENTINEL:
            break
        path = item
        if not os.path.exists(path):
            continue
        # FIX: проверяем stop перед загрузкой, чтобы не начать воспроизведение
        # после того как пришёл stop (например, при перебивании во время генерации)
        if _should_stop(stop_ev):
            break
        try:
            _ensure_mixer()
            if not pygame.mixer.get_init():
                continue
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            _set_playing(True)
            while pygame.mixer.music.get_busy():
                if _should_stop(stop_ev):
                    _release()
                    break
                time.sleep(0.05)
            _release()
            _set_playing(False)
        except Exception as e:
            _set_playing(False)
            _release()
            print(f"[TTS] playback error: {e}")
    _release()
    _set_playing(False)


def _listen_thread(audio_core, stop_ev, interrupt_ev, audio_box):
    from voice.turn import TurnManager
    # FIX: импортируем CHUNK_SIZE из core.config, а не из voice.audio_core
    # (в audio_core нет публичного экспорта CHUNK_SIZE)
    tap = audio_core.create_tap()
    fired = threading.Event()

    def _iter():
        while not interrupt_ev.is_set() and not (stop_ev and stop_ev.is_set()):
            try:
                item = tap.get(timeout=0.2)
                # FIX (audit 3): сравниваем с sentinel из audio_core,
                # т.к. tap создан им и его же remove_tap() кладёт sentinel.
                if item is _AUDIO_STOP_SENTINEL:
                    return
                yield item
            except Exception:
                return

    def _vad_iter():
        for chunk in _iter():
            try:
                from voice.stt import vad_prob
                p = vad_prob(chunk)
                if p >= TURN_VAD_TRIGGER and not fired.is_set():
                    fired.set()
                    _set_stop(True)
                    _release()
                    print("[TTS] Перебивание обнаружено")
            except Exception:
                pass
            yield chunk

    try:
        tm = TurnManager(chunk_size=CHUNK_SIZE)
        audio = tm.collect_utterance(_vad_iter(), stop_event=stop_ev)
        if audio is not None and len(audio) > 0:
            audio_box["audio"] = audio
            interrupt_ev.set()
    finally:
        audio_core.remove_tap(tap)


def _run(text: str, stop_ev=None, allow_interrupt=False, audio_core=None):
    if not text or not text.strip():
        return None

    _set_stop(False)
    _set_playing(False)
    session_dir = _new_session_dir()
    # FIX: _cleanup_orphans вызывается один раз при старте сессии,
    # не нужно вызывать его повторно при каждом чанке
    _cleanup_orphans(keep=session_dir)
    _ensure_mixer()

    q = Queue(maxsize=2)
    interrupt_ev = threading.Event()
    audio_box = {"audio": None}

    t_gen  = threading.Thread(target=_gen_thread,  args=(text, session_dir, q, stop_ev), daemon=True)
    t_play = threading.Thread(target=_play_thread, args=(q, stop_ev), daemon=True)
    t_lst  = None
    if allow_interrupt and audio_core is not None:
        t_lst = threading.Thread(
            target=_listen_thread,
            args=(audio_core, stop_ev, interrupt_ev, audio_box),
            daemon=True,
        )

    t_gen.start(); t_play.start()
    if t_lst: t_lst.start()

    try:
        while True:
            done = not t_gen.is_alive() and not t_play.is_alive()
            lst_done = t_lst is None or not t_lst.is_alive()

            if _get_stop() and t_lst is not None:
                try: q.put_nowait(_TTS_STOP_SENTINEL)
                except Exception: pass
                t_gen.join(timeout=0.5); t_play.join(timeout=0.5)
                _release(); _set_playing(False)
                if not lst_done:
                    t_lst.join(timeout=_LISTENER_WAIT_TIMEOUT)
                return audio_box["audio"]

            if stop_ev is not None and stop_ev.is_set():
                _set_stop(True)
                try: q.put_nowait(_TTS_STOP_SENTINEL)
                except Exception: pass
                _release()
                t_gen.join(timeout=0.1); t_play.join(timeout=0.1)
                if t_lst: t_lst.join(timeout=0.1)
                _set_playing(False)
                return None

            if done and lst_done:
                _set_playing(False); _release()
                return audio_box["audio"]

            time.sleep(0.05)
    finally:
        _cleanup(session_dir)


def say(text: str, stop_event=None):
    _run(text, stop_ev=stop_event)


def speak_and_handle(text: str, stop_event=None, audio_core=None):
    return _run(text, stop_ev=stop_event, allow_interrupt=True, audio_core=audio_core)


def stop_speaking():
    _set_stop(True)
    _release()


def is_speaking() -> bool:
    with _state_lock:
        return _playing
