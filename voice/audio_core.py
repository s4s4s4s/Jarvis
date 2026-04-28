import queue
import threading
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from core.config import SAMPLE_RATE_MIC, CHUNK_SIZE, PRE_ROLL_SEC, MIC_DEVICE

_STOP_SENTINEL = object()

# ~6 секунд буфера при 16 кГц / chunk 512: 16000/512*6 ≈ 188 чанков
_TAP_QUEUE_MAXSIZE = 200


class AudioCore:
    def __init__(self):
        self._taps: list[queue.Queue] = []
        self._taps_lock = threading.Lock()
        self._pre_roll_len = max(1, int(PRE_ROLL_SEC * SAMPLE_RATE_MIC / CHUNK_SIZE))
        self._pre_roll: deque = deque(maxlen=self._pre_roll_len)
        self._stream = None
        self._stop_requested = False
        self._dropped_chunks = 0

    def create_tap(self, pre_roll: bool = False) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_TAP_QUEUE_MAXSIZE)
        with self._taps_lock:
            self._taps.append(q)
            if pre_roll and self._pre_roll:
                for chunk in self._pre_roll:
                    try:
                        q.put_nowait(chunk)
                    except queue.Full:
                        pass
        return q

    def remove_tap(self, q: queue.Queue) -> None:
        with self._taps_lock:
            try:
                self._taps.remove(q)
            except ValueError:
                pass
        try:
            q.put_nowait(_STOP_SENTINEL)
        except Exception:
            pass

    def stream_chunks(self, stop_event: Optional[threading.Event] = None):
        tap = self.create_tap(pre_roll=False)
        try:
            while True:
                if self._stop_requested:
                    return
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    item = tap.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _STOP_SENTINEL:
                    return
                yield item
        finally:
            self.remove_tap(tap)

    def get_pre_roll(self) -> list:
        with self._taps_lock:
            return list(self._pre_roll)

    def get_dropped_chunks(self) -> int:
        return self._dropped_chunks

    def request_stop(self) -> None:
        self._stop_requested = True
        self.stop()

    def _callback(self, indata: np.ndarray, frames, time_, status):
        chunk = indata[:, 0].copy()
        self._pre_roll.append(chunk)
        with self._taps_lock:
            dead = []
            for q in self._taps:
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    self._dropped_chunks += 1
                except Exception:
                    dead.append(q)
            for q in dead:
                try:
                    self._taps.remove(q)
                except ValueError:
                    pass

    def start(self):
        self._stop_requested = False
        self._dropped_chunks = 0

        # Resolve device: prefer configured MIC_DEVICE, fallback to system default.
        # WASAPI devices give cleaner signal than MME on Windows (no audio enhancements).
        device = MIC_DEVICE
        if device is not None:
            try:
                info = sd.query_devices(device)
                if info["max_input_channels"] < 1:
                    print(f"[AudioCore] WARNING: device {device} has no input channels, falling back to default")
                    device = None
                else:
                    print(f"[AudioCore] Using mic: [{device}] {info['name']}")
            except Exception as e:
                print(f"[AudioCore] WARNING: cannot query device {device}: {e}, falling back to default")
                device = None

        self._stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE_MIC,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._taps_lock:
            for q in self._taps:
                try:
                    q.put_nowait(_STOP_SENTINEL)
                except Exception:
                    pass
            self._taps.clear()
