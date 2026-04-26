# voice/wake.py
import threading
import time
from collections import deque

import numpy as np

from core.config import (
    SAMPLE_RATE_MIC,
    CHUNK_SIZE,
    WAKE_CHECK_SEC,
    WAKE_MIN_CHECK_INTERVAL_SEC,
    WAKE_FAIL_COOLDOWN_SEC,
    WAKE_SUCCESS_COOLDOWN_SEC,
    WAKE_VAD_TRIGGER,
    WAKE_VAD_HOLD,
    WAKE_MIN_SPEECH_CHUNKS,
    WAKE_MAX_SILENCE_CHUNKS,
    WAKE_MAX_TEXT_LEN,
    WAKE_PHRASES,
    WAKE_BLOCKLIST,
    IGNORE_PHRASES,
)
from .stt import transcribe, vad_prob

BAD_TAILS = [
    "субтитры сделал dimatorzok",
    "субтитры делал dimatorzok",
    "субтитры подготовил dimatorzok",
    "subtitles by dimatorzok",
]


def clean_weird_tail(text: str) -> str:
    t_low = text.lower()
    for tail in BAD_TAILS:
        idx = t_low.rfind(tail)
        if idx != -1:
            return text[:idx].strip()
    return text


class WakeDetector:
    def __init__(self):
        self.window_chunks = max(1, int((WAKE_CHECK_SEC * SAMPLE_RATE_MIC) / CHUNK_SIZE))
        self.window = deque(maxlen=self.window_chunks)
        self.last_check_ts = 0.0
        self.cooldown_until = 0.0
        self.speech_streak = 0
        self.silence_streak = 0
        self.in_speech = False
        self.last_vad = 0.0

        # FIX: transcribe запускается в отдельном потоке, чтобы не блокировать
        # основной аудио-цикл на время инференса Whisper (~300-500 мс)
        self._check_lock = threading.Lock()
        self._check_thread: threading.Thread | None = None
        self._pending_result: bool | None = None  # True = wake найден
        self._fired = False  # True = wake уже зафиксирован, ждём consume

    def reset(self):
        self.speech_streak = 0
        self.silence_streak = 0
        self.in_speech = False
        self.last_vad = 0.0
        self._fired = False

    def contains_wake_word(self, text: str) -> bool:
        t = text.lower().strip()
        if any(bad in t for bad in WAKE_BLOCKLIST):
            return False
        return any(p in t for p in WAKE_PHRASES)

    def _update_vad_state(self, prob: float):
        self.last_vad = prob
        if not self.in_speech:
            if prob >= WAKE_VAD_TRIGGER:
                self.speech_streak += 1
            else:
                self.speech_streak = 0
            if self.speech_streak >= WAKE_MIN_SPEECH_CHUNKS:
                self.in_speech = True
                self.silence_streak = 0
        else:
            if prob >= WAKE_VAD_HOLD:
                self.silence_streak = 0
            else:
                self.silence_streak += 1
                if self.silence_streak >= WAKE_MAX_SILENCE_CHUNKS:
                    self.reset()

    def _run_transcribe(self, audio: np.ndarray, now: float):
        """Запускается в отдельном потоке — не блокирует аудио-цикл."""
        text = transcribe(audio, log=False)
        if not text:
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            with self._check_lock:
                self._check_thread = None
            return
        text = clean_weird_tail(text)
        t_low = text.lower().strip()
        if any(p in t_low for p in IGNORE_PHRASES):
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            with self._check_lock:
                self._check_thread = None
            return
        if len(t_low) > WAKE_MAX_TEXT_LEN:
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            with self._check_lock:
                self._check_thread = None
            return
        print(
            f"\n[wake-check vad={self.last_vad:.2f} in_speech={self.in_speech} "
            f"speech_streak={self.speech_streak}] {text}"
        )
        if self.contains_wake_word(t_low):
            with self._check_lock:
                self._pending_result = True
                self._check_thread = None
        else:
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            with self._check_lock:
                self._check_thread = None

    def process_chunk(self, chunk) -> bool:
        self.window.append(chunk)
        if len(self.window) < self.window.maxlen:
            return False

        # FIX: если transcribe-поток нашёл wake-word, возвращаем True
        with self._check_lock:
            if self._pending_result:
                self._pending_result = None
                self.reset()
                self.cooldown_until = time.time() + WAKE_SUCCESS_COOLDOWN_SEC
                return True

        now = time.time()
        if now < self.cooldown_until:
            return False

        prob = vad_prob(chunk)
        self._update_vad_state(prob)
        if not self.in_speech:
            return False
        if now - self.last_check_ts < WAKE_MIN_CHECK_INTERVAL_SEC:
            return False

        with self._check_lock:
            if self._check_thread is not None:
                # Предыдущий transcribe ещё не завершился — пропускаем этот чек
                return False

        self.last_check_ts = now
        audio = np.concatenate(list(self.window))

        # FIX: запускаем transcribe в daemon-потоке, не блокируем аудио-цикл
        t = threading.Thread(
            target=self._run_transcribe,
            args=(audio, now),
            daemon=True,
            name="wake-transcribe",
        )
        with self._check_lock:
            self._check_thread = t
        t.start()
        return False
