# ui/bridge.py
from __future__ import annotations

import sys
import threading
import traceback
from typing import Callable, Optional, Any

from PySide6.QtCore import QObject, Signal, Slot

from voice.state import AssistantState  # noqa: F401  (re-exported for callers)

_CALLBACK_ALIASES = {
    "on_state": "state",        "on_status": "state",       "state_changed": "state",
    "on_user_text": "user_text", "user_text": "user_text",
    "on_assistant_text": "assistant_text", "on_assistant": "assistant_text", "assistant_text": "assistant_text",
    "on_system_log": "system_log", "on_log": "system_log",  "system_log": "system_log",
    "on_error": "error",        "error": "error",
}


class _LineBuffer:
    """Буферизует текст и вызывает callback построчно."""
    def __init__(self, callback: Callable[[str], None]):
        self._cb = callback
        self._buf = ""

    def feed(self, data: str) -> None:
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                try:
                    self._cb(line)
                except Exception:
                    pass


class _StreamRedirector:
    """
    Перехватывает один stream (stdout ИЛИ stderr).
    Пишет в original И шлёт построчно в callback.
    """
    def __init__(self, original, callback: Callable[[str], None]):
        self._original = original
        self._lb = _LineBuffer(callback)

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            if self._original is not None:
                try:
                    self._original.write(data)
                except Exception:
                    pass
            self._lb.feed(data)
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass

    def fileno(self):
        if self._original is not None:
            return self._original.fileno()
        raise OSError("no fileno")


class JarvisBridge(QObject):
    state_changed  = Signal(object)
    user_text      = Signal(str)
    assistant_text = Signal(str)
    system_log     = Signal(str)
    error          = Signal(str)

    def __init__(self, parent: Optional[QObject] = None, **kwargs: Any) -> None:
        qt_parent = parent if parent is not None else kwargs.pop("parent", None)
        super().__init__(qt_parent)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._orig_stdout = None
        self._orig_stderr = None

        self._external_callbacks: dict[str, list[Callable[..., None]]] = {
            k: [] for k in ("state", "user_text", "assistant_text", "system_log", "error")
        }
        for key, value in list(kwargs.items()):
            if value is None:
                continue
            channel = _CALLBACK_ALIASES.get(key)
            if channel and callable(value):
                self._external_callbacks[channel].append(value)

    # ---- Public API ----

    def is_running(self) -> bool:
        t = self._thread
        return self._started and t is not None and t.is_alive()

    def running(self) -> bool:
        return self.is_running()

    def start(self) -> None:
        if self.is_running():
            return
        self._started = True
        self._stop_event.clear()
        self._install_redirect()   # stdout/stderr → system_log ОДИН раз
        self._thread = threading.Thread(
            target=self._run_assistant,
            name="JarvisAssistantThread",
            daemon=True,
        )
        self._thread.start()
        # НЕ используем print здесь — redirect ещё не читается GUI
        try:
            self.system_log.emit("[bridge] assistant thread started\n")
        except Exception:
            pass

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._restore_redirect()
        self._started = False
        self._thread = None

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop(timeout=timeout)

    # ---- Emitters ----

    def emit_state(self, state) -> None:
        try:
            self.state_changed.emit(state)
        except Exception:
            pass
        self._call_external("state", state)

    def emit_user_text(self, text: str) -> None:
        if not text:
            return
        try:
            self.user_text.emit(str(text))
        except Exception:
            pass
        self._call_external("user_text", str(text))

    def emit_assistant_text(self, text: str) -> None:
        if not text:
            return
        try:
            self.assistant_text.emit(str(text))
        except Exception:
            pass
        self._call_external("assistant_text", str(text))

    # backward-compat
    def on_state(self, s):          self.emit_state(s)
    def on_status(self, s):         self.emit_state(s)
    def on_user_text(self, t):      self.emit_user_text(t)
    def on_assistant_text(self, t): self.emit_assistant_text(t)
    def start_assistant(self):      self.start()
    def stop_assistant(self, timeout=5.0): self.stop(timeout)

    # ---- Internal ----

    def _sys_log(self, line: str) -> None:
        """Одна строка лога → system_log signal. Вызывается из StreamRedirector."""
        if not line:
            return
        msg = line.rstrip("\r\n") + "\n"
        try:
            self.system_log.emit(msg)
        except Exception:
            pass
        for cb in self._external_callbacks.get("system_log", []):
            try:
                cb(msg)
            except Exception:
                pass

    def _call_external(self, channel: str, *args) -> None:
        for cb in self._external_callbacks.get(channel, []):
            try:
                cb(*args)
            except Exception:
                pass

    def _install_redirect(self) -> None:
        if self._orig_stdout is not None:
            return  # уже установлен
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        # stdout и stderr используют ОДИН И ТОТ ЖЕ _sys_log callback
        # но это один объект — нет дублей
        sys.stdout = _StreamRedirector(self._orig_stdout, self._sys_log)  # type: ignore
        # stderr намеренно НЕ редиректим — pygame/warnings пишут туда мусор
        # и они не нужны в GUI-логе

    def _restore_redirect(self) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        # stderr не трогали — не восстанавливаем

    def _run_assistant(self) -> None:
        try:
            from voice import assistant as assistant_mod
        except Exception as e:
            err = f"[bridge] import error: {e}\n{traceback.format_exc()}"
            print(err)
            try:
                self.error.emit(err)
            except Exception:
                pass
            return

        main_fn = getattr(assistant_mod, "main", None)
        if main_fn is None:
            print("[bridge] voice.assistant.main not found")
            return

        try:
            main_fn(
                stop_event=self._stop_event,
                on_state=self.emit_state,
                on_user_text=self.emit_user_text,
                on_assistant_text=self.emit_assistant_text,
                # on_system_log намеренно НЕ передаём — логи идут через stdout
            )
        except Exception as e:
            print(f"[bridge] assistant crashed: {e}\n{traceback.format_exc()}")
            try:
                self.error.emit(str(e))
            except Exception:
                pass

        print("[bridge] assistant thread exiting")
        # FIX (audit 3): восстанавливаем stdout, если ассистент завершился сам
        # (напр. Ollama недоступна). Иначе stop() не вызывается → редирект
        # остаётся висеть, и повторный start() не установит новый (из-за
        # проверки `if self._orig_stdout is not None: return`).
        self._started = False
        self._restore_redirect()

    @Slot()
    def slot_start(self) -> None:
        self.start()

    @Slot()
    def slot_stop(self) -> None:
        self.stop()
