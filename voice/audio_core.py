# voice/audio_core.py

import queue
import threading
import time
from collections import deque
from typing import Optional

import sounddevice as sd

from .config import SAMPLE_RATE_MIC, CHUNK_SIZE, PRE_ROLL_SEC


_STOP_SENTINEL = object()


class AudioCore:
    def __init__(self):
        self.q = queue.Queue()
        self.pre_roll_chunks = int((PRE_ROLL_SEC * SAMPLE_RATE_MIC) / CHUNK_SIZE)
        self.ring = deque(maxlen=self.pre_roll_chunks)

        self._tap_lock = threading.Lock()
        self._tap_queues: list[queue.Queue] = []

    def callback(self, indata, frames, time_info, status):
        chunk = indata.copy().flatten()

        self.ring.append(chunk)
        self.q.put(chunk)

        with self._tap_lock:
            dead = []
            for tap_q in self._tap_queues:
                try:
                    tap_q.put_nowait(chunk)
                except Exception:
                    dead.append(tap_q)

            if dead:
                self._tap_queues = [q for q in self._tap_queues if q not in dead]

    def request_stop(self):
        try:
            self.q.put_nowait(_STOP_SENTINEL)
        except Exception:
            pass

        with self._tap_lock:
            for tap_q in self._tap_queues:
                try:
                    tap_q.put_nowait(_STOP_SENTINEL)
                except Exception:
                    pass

    def clear_queue(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Exception:
                break

    def create_tap(self) -> queue.Queue:
        tap_q = queue.Queue(maxsize=256)
        with self._tap_lock:
            self._tap_queues.append(tap_q)
        return tap_q

    def remove_tap(self, tap_q: queue.Queue):
        with self._tap_lock:
            self._tap_queues = [q for q in self._tap_queues if q is not tap_q]

    def stream_chunks(self, stop_event: Optional[object] = None):
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE_MIC,
                    channels=1,
                    dtype="float32",
                    blocksize=CHUNK_SIZE,
                    callback=self.callback,
                ):
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            return

                        try:
                            item = self.q.get(timeout=0.20)
                        except queue.Empty:
                            continue

                        if item is _STOP_SENTINEL:
                            return

                        yield item

            except Exception as e:
                print(f"[audio_core] Поток упал: {e}, перезапускаю...")
                self.clear_queue()
                time.sleep(0.5)

                if stop_event is not None and stop_event.is_set():
                    return

    def get_pre_roll(self):
        return list(self.ring)
