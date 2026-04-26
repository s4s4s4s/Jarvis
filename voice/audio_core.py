# voice/audio_core.py
import queue
import threading
import numpy as np
import sounddevice as sd

from core.config import SAMPLE_RATE_MIC, CHUNK_SIZE, PRE_ROLL_SEC

_STOP_SENTINEL = object()


class AudioCore:
    def __init__(self):
        self._taps: list[queue.Queue] = []
        self._taps_lock = threading.Lock()
        self._pre_roll_len = int(PRE_ROLL_SEC * SAMPLE_RATE_MIC / CHUNK_SIZE)
        self._pre_roll: list = []
        self._stream = None

    def create_tap(self, pre_roll: bool = False) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._taps_lock:
            self._taps.append(q)
            if pre_roll and self._pre_roll:
                for chunk in self._pre_roll:
                    q.put(chunk)
        return q

    def remove_tap(self, q: queue.Queue) -> None:
        with self._taps_lock:
            try:
                self._taps.remove(q)
            except ValueError:
                pass
        q.put(_STOP_SENTINEL)

    def _callback(self, indata: np.ndarray, frames, time_, status):
        chunk = indata[:, 0].copy()
        self._pre_roll.append(chunk)
        if len(self._pre_roll) > self._pre_roll_len:
            self._pre_roll.pop(0)
        with self._taps_lock:
            for q in self._taps:
                q.put(chunk)

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE_MIC,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._taps_lock:
            for q in self._taps:
                q.put(_STOP_SENTINEL)
            self._taps.clear()