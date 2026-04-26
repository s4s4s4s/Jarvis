# voice/turn.py
import time
from typing import Optional

import numpy as np

from core.config import (
    SILENCE_MS,
    MAX_RECORD_SEC,
    MIN_UTTERANCE_SEC,
    SAMPLE_RATE_MIC,
    TURN_VAD_TRIGGER,
    TURN_VAD_HOLD,
)
from .stt import vad_prob


class TurnManager:
    def __init__(self, chunk_size):
        self.chunk_ms = int(chunk_size / SAMPLE_RATE_MIC * 1000)
        self.silence_chunks_needed = max(1, int(SILENCE_MS / self.chunk_ms))
        self.max_chunks = max(1, int(MAX_RECORD_SEC * 1000 / self.chunk_ms))
        self.min_samples = int(MIN_UTTERANCE_SEC * SAMPLE_RATE_MIC)

    def _should_stop(self, stop_event: Optional[object]) -> bool:
        return stop_event is not None and stop_event.is_set()

    def _next_chunk(self, chunk_iter, stop_event: Optional[object] = None):
        if self._should_stop(stop_event):
            return None
        try:
            return next(chunk_iter)
        except StopIteration:
            return None

    def collect_utterance(self, chunk_iter, initial_frames=None, stop_event: Optional[object] = None):
        frames = list(initial_frames) if initial_frames else []
        silence_counter = 0
        speech_started = False

        for _ in range(self.max_chunks):
            if self._should_stop(stop_event):
                return None
            chunk = self._next_chunk(chunk_iter, stop_event=stop_event)
            if chunk is None:
                return None
            prob = vad_prob(chunk)
            if prob >= TURN_VAD_TRIGGER:
                speech_started = True
                silence_counter = 0
                frames.append(chunk)
            elif speech_started:
                frames.append(chunk)
                if prob < TURN_VAD_HOLD:
                    silence_counter += 1
                    if silence_counter >= self.silence_chunks_needed:
                        break
                else:
                    silence_counter = 0

        if not speech_started or not frames:
            return None
        audio = np.concatenate(frames)
        if len(audio) < self.min_samples:
            return None
        return audio

    def collect_with_timeout(
        self,
        chunk_iter,
        idle_timeout_sec: float = 5.0,
        stop_event: Optional[object] = None,
    ):
        frames = []
        silence_counter = 0
        speech_started = False
        idle_start = time.time()

        for _ in range(self.max_chunks):
            if self._should_stop(stop_event):
                return None
            chunk = self._next_chunk(chunk_iter, stop_event=stop_event)
            if chunk is None:
                return None
            prob = vad_prob(chunk)
            if prob >= TURN_VAD_TRIGGER:
                speech_started = True
                idle_start = time.time()
                silence_counter = 0
                frames.append(chunk)
            else:
                if speech_started:
                    frames.append(chunk)
                    if prob < TURN_VAD_HOLD:
                        silence_counter += 1
                        if silence_counter >= self.silence_chunks_needed:
                            break
                    else:
                        silence_counter = 0
                else:
                    if time.time() - idle_start >= idle_timeout_sec:
                        return None

        if not frames:
            return None
        audio = np.concatenate(frames)
        if len(audio) < self.min_samples:
            return None
        return audio