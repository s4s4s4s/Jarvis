# C:\jarvis\ui\bridge.py
# -*- coding: utf-8 -*-
"""
JarvisBridge — мост между PySide6 GUI и голосовым движком (voice.assistant).
"""

from __future__ import annotations

import sys
import threading
import traceback
from typing import Callable, Optional, Any

from PySide6.QtCore import QObject, Signal, Slot

try:
    from voice.state import AssistantState
except Exception:
    AssistantState = None  # type: ignore


_CALLBACK_ALIASES = {
    "on_state": "state",
    "on_status": "state",
    "state_changed": "state",

    "on_user_text": "user_text",
    "user_text": "user_text",

    "on_assistant_text": "assistant_text",
    "on_assistant": "assistant_text",
    "assistant_text": "assistant_text",

    "on_system_log": "system_log",
    "on_log": "system_log",
    "system_log": "system_log",

    "on_error": "error",
    "error": "error",
}


class _StreamRedirector:
    def __init__(self, original, callback: Callable[[str], None]):
        self._original = original
        self._callback = callback
        self._buffer = ""

    def write(self, data: str) -> int:
        try:
            if self._original is not None:
                try:
                    self._original.write(data)
                except Exception:
                    pass
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line:
                    try:
                        self._callback(line)
                    except Exception:
                        pass
        except Exception:
            pass
        return len(data) if data else 0

    def flush(self) -> None:
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass


class JarvisBridge(QObject):
    state_changed = Signal(object)
    user_text = Signal(str)
    assistant_text = Signal(str)
    system_log = Signal(str)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None, **kwargs: Any) -> None:
        qt_parent = parent
        if qt_parent is None:
            qt_parent = kwargs.pop("parent", None)
        super().__init__(qt_parent)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        self._orig_stdout = None
        self._orig_stderr = None
        self._stdout_redirect: Optional[_StreamRedirector] = None
        self._stderr_redirect: Optional[_StreamRedirector] = None

        self._external_callbacks: dict[str, list[Callable[..., None]]] = {
            "state": [],
            "user_text": [],
            "assistant_text": [],
            "system_log": [],
            "error": [],
        }

        for key, value in list(kwargs.items()):
            if value is None:
                continue
            channel = _CALLBACK_ALIASES.get(key)
            if channel is None:
                self._emit_system(f"[bridge] unknown kwarg ignored: {key}")
                continue
            if not callable(value):
                self._emit_system(f"[bridge] kwarg {key} is not callable, ignored")
                continue
            self._external_callbacks[channel].append(value)

    # ---------- Публичное API ----------

    def is_running(self) -> bool:
        if not self._started:
            return False
        t = self._thread
        return t is not None and t.is_alive()

    def running(self) -> bool:
        return self.is_running()

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self.is_running():
            self._emit_system("[bridge] assistant already running")
            return
        self._started = True
        self._stop_event.clear()
        # self._install_stream_redirect()   # ← ЗАКОММЕНТИРОВАТЬ

        self._thread = threading.Thread(
            target=self._run_assistant,
            name="JarvisAssistantThread",
            daemon=True,
        )
        self._thread.start()
        self._emit_system("[bridge] assistant thread started")


    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._emit_system("[bridge] stopping assistant…")
        self._stop_event.set()

        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                self._emit_system("[bridge] assistant thread did not stop in time")
            else:
                self._emit_system("[bridge] assistant thread stopped")

        self._restore_stream_redirect()
        self._started = False
        self._thread = None

    def start_assistant(self) -> None:
        self.start()

    def stop_assistant(self, timeout: float = 5.0) -> None:
        self.stop(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop(timeout=timeout)

    # ---------- Эмиттеры ----------

    def emit_state(self, state) -> None:
        try:
            self.state_changed.emit(state)
        except Exception as e:
            self._emit_system(f"[bridge] emit_state error: {e}")
        self._call_external("state", state)

    def emit_user_text(self, text: str) -> None:
        if not text:
            return
        try:
            self.user_text.emit(str(text))
        except Exception as e:
            self._emit_system(f"[bridge] emit_user_text error: {e}")
        self._call_external("user_text", str(text))

    def emit_assistant_text(self, text: str) -> None:
        if not text:
            return
        try:
            self.assistant_text.emit(str(text))
        except Exception as e:
            self._emit_system(f"[bridge] emit_assistant_text error: {e}")
        self._call_external("assistant_text", str(text))

    def emit_system_log(self, text: str) -> None:
        self._emit_system(str(text))

    def on_state(self, state) -> None:
        self.emit_state(state)

    def on_status(self, state) -> None:
        self.emit_state(state)

    def on_user_text(self, text: str) -> None:
        self.emit_user_text(text)

    def on_assistant_text(self, text: str) -> None:
        self.emit_assistant_text(text)

    def on_system_log(self, text: str) -> None:
        self.emit_system_log(text)

    # ---------- Внутреннее ----------

    def _call_external(self, channel: str, *args) -> None:
        for cb in self._external_callbacks.get(channel, []):
            try:
                cb(*args)
            except Exception as e:
                try:
                    self.system_log.emit(
                        f"[bridge] external {channel} callback error: {e}"
                    )
                except Exception:
                    pass

    def _emit_system(self, text: str) -> None:
        msg = str(text).rstrip("\r\n")
        if not msg:
            return
        msg = msg + "\n"
        try:
            self.system_log.emit(msg)
        except Exception:
            pass
        for cb in self._external_callbacks.get("system_log", []):
            try:
                cb(msg)
            except Exception:
                pass


    def _install_stream_redirect(self) -> None:
        try:
            if self._orig_stdout is None:
                self._orig_stdout = sys.stdout
                self._stdout_redirect = _StreamRedirector(sys.stdout, self._emit_system)
                sys.stdout = self._stdout_redirect  # type: ignore[assignment]
            if self._orig_stderr is None:
                self._orig_stderr = sys.stderr
                self._stderr_redirect = _StreamRedirector(sys.stderr, self._emit_system)
                sys.stderr = self._stderr_redirect  # type: ignore[assignment]
        except Exception as e:
            self._emit_system(f"[bridge] stream redirect install error: {e}")

    def _restore_stream_redirect(self) -> None:
        try:
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
                self._orig_stdout = None
                self._stdout_redirect = None
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
                self._orig_stderr = None
                self._stderr_redirect = None
        except Exception:
            pass

    def _run_assistant(self) -> None:
        try:
            from voice import assistant as assistant_mod
        except Exception as e:
            err = f"[bridge] failed to import voice.assistant: {e}\n{traceback.format_exc()}"
            self._emit_system(err)
            try:
                self.error.emit(err)
            except Exception:
                pass
            self._call_external("error", err)
            return

        main_fn = getattr(assistant_mod, "main", None)
        if main_fn is None:
            self._emit_system("[bridge] voice.assistant.main not found")
            return

        try:
            main_fn(
                stop_event=self._stop_event,
                on_state=self.emit_state,
                on_user_text=self.emit_user_text,
                on_assistant_text=self.emit_assistant_text,
                on_system_log=self.emit_system_log,
            )
        except Exception as e:
            self._emit_system(
                f"[bridge] assistant main crashed: {e}\n{traceback.format_exc()}"
            )
            try:
                self.error.emit(str(e))
            except Exception:
                pass

        self._emit_system("[bridge] assistant thread exiting")
        self._started = False

    @Slot()
    def slot_start(self) -> None:
        self.start()

    @Slot()
    def slot_stop(self) -> None:
        self.stop()
