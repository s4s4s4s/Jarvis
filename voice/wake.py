# voice/wake.py
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
    IGNORE_PHRASES,  # единый источник истины из core.config
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

    def reset(self):
        self.speech_streak = 0
        self.silence_streak = 0
        self.in_speech = False
        self.last_vad = 0.0

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

    def process_chunk(self, chunk) -> bool:
        self.window.append(chunk)
        if len(self.window) < self.window.maxlen:
            return False
        now = time.time()
        if now < self.cooldown_until:
            return False
        prob = vad_prob(chunk)
        self._update_vad_state(prob)
        if not self.in_speech:
            return False
        if now - self.last_check_ts < WAKE_MIN_CHECK_INTERVAL_SEC:
            return False
        self.last_check_ts = now
        audio = np.concatenate(list(self.window))
        text = transcribe(audio, log=False)
        if not text:
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            return False
        text = clean_weird_tail(text)
        t_low = text.lower().strip()
        if any(p in t_low for p in IGNORE_PHRASES):
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            return False
        if len(t_low) > WAKE_MAX_TEXT_LEN:
            self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
            return False
        print(
            f"\n[wake-check vad={prob:.2f} in_speech={self.in_speech} "
            f"speech_streak={self.speech_streak}] {text}"
        )
        if self.contains_wake_word(t_low):
            self.reset()
            self.cooldown_until = now + WAKE_SUCCESS_COOLDOWN_SEC
            return True
        self.cooldown_until = now + WAKE_FAIL_COOLDOWN_SEC
        return False
