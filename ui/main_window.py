# ui/main_window.py

import sys
import threading
import traceback
from io import StringIO

from PySide6.QtCore import QTimer, Qt, QObject, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .bridge import JarvisBridge

try:
    from voice.state import AssistantState
except Exception:
    AssistantState = None  # type: ignore


class _StdoutSignaler(QObject):
    line = Signal(str)


class StreamRedirector(StringIO):
    def __init__(self, signaler: _StdoutSignaler, mirror=None):
        super().__init__()
        self._signaler = signaler
        self._mirror = mirror

    def write(self, text):
        if not text:
            return 0
        try:
            if self._mirror is not None:
                try:
                    self._mirror.write(text)
                except Exception:
                    pass
            self._signaler.line.emit(str(text))
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            if self._mirror is not None:
                self._mirror.flush()
        except Exception:
            pass


class JarvisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jarvis Control Center")
        self.resize(1320, 860)

        # Создаём bridge БЕЗ автозапуска
        self.bridge = JarvisBridge()
        self.bridge.user_text.connect(self._on_user_text, Qt.QueuedConnection)
        self.bridge.assistant_text.connect(self._on_assistant_text, Qt.QueuedConnection)
        self.bridge.system_log.connect(self.on_system_log, Qt.QueuedConnection)
        self.bridge.state_changed.connect(self.on_state_changed, Qt.QueuedConnection)
        self.bridge.error.connect(self.on_error, Qt.QueuedConnection)

        self._stdout_signaler = _StdoutSignaler(self)
        self._stdout_signaler.line.connect(self.on_stdout_line, Qt.QueuedConnection)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.system_buffer = []
        self.dialog_buffer = []

        self.build_ui()

        self.flush_timer = QTimer(self)
        self.flush_timer.timeout.connect(self.flush_buffers)
        self.flush_timer.start(100)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QFrame()
        header.setObjectName("Card")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)

        title_box = QVBoxLayout()
        title = QLabel("JARVIS")
        title.setObjectName("Title")
        subtitle = QLabel("Голосовой ассистент Jarvis — STT / LLM / TTS / GUI")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.status_label = QLabel("Отключён")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("StatusIdle")
        self.status_label.setMinimumWidth(190)
        self.status_label.setFixedHeight(38)

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)

        controls = QFrame()
        controls.setObjectName("Card")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(20, 18, 20, 18)
        controls_layout.setSpacing(12)

        self.start_btn = QPushButton("Запустить Jarvis")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.start_assistant)

        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_assistant)

        self.clear_btn = QPushButton("Очистить логи")
        self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.clicked.connect(self.clear_logs)

        self.exit_btn = QPushButton("Выход")
        self.exit_btn.setObjectName("SecondaryButton")
        self.exit_btn.clicked.connect(self.close)

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addWidget(self.exit_btn)
        controls_layout.addStretch()

        grid = QGridLayout()
        grid.setSpacing(16)

        dialog_card = QFrame()
        dialog_card.setObjectName("Card")
        dialog_layout = QVBoxLayout(dialog_card)
        dialog_layout.setContentsMargins(18, 18, 18, 18)
        dialog_layout.setSpacing(10)
        dialog_title = QLabel("Диалог")
        dialog_title.setObjectName("SectionTitle")
        self.dialog_view = QTextEdit()
        self.dialog_view.setObjectName("DialogBox")
        self.dialog_view.setReadOnly(True)
        dialog_layout.addWidget(dialog_title)
        dialog_layout.addWidget(self.dialog_view)

        system_card = QFrame()
        system_card.setObjectName("Card")
        system_layout = QVBoxLayout(system_card)
        system_layout.setContentsMargins(18, 18, 18, 18)
        system_layout.setSpacing(10)
        system_title = QLabel("Системный лог")
        system_title.setObjectName("SectionTitle")
        self.system_view = QTextEdit()
        self.system_view.setObjectName("SystemBox")
        self.system_view.setReadOnly(True)
        system_layout.addWidget(system_title)
        system_layout.addWidget(self.system_view)

        info_card = QFrame()
        info_card.setObjectName("Card")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(10)
        info_title = QLabel("Справка")
        info_title.setObjectName("SectionTitle")
        self.info_view = QTextEdit()
        self.info_view.setObjectName("InfoBox")
        self.info_view.setReadOnly(True)
        self.info_view.setPlainText(
            "Jarvis — голосовой ассистент\n"
            "TTS — streaming Edge TTS\n"
            "STT — Whisper large-v3\n"
            "LLM — Ollama (router + fast + heavy)\n"
            "Память — долгосрочная + история сессий"
        )
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_view)

        grid.addWidget(dialog_card, 0, 0, 2, 2)
        grid.addWidget(system_card, 0, 2, 1, 1)
        grid.addWidget(info_card, 1, 2, 1, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)

        layout.addWidget(header)
        layout.addWidget(controls)
        layout.addLayout(grid)

        self.setStyleSheet("""
            QWidget {
                background: #0f1115;
                color: #e8ecf3;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
            }
            QMainWindow { background: #0f1115; }
            QFrame#Card {
                background: #171a21;
                border: 1px solid #252a35;
                border-radius: 16px;
            }
            QLabel#Title {
                font-size: 30px;
                font-weight: 800;
                color: #f4f7fb;
                letter-spacing: 1px;
            }
            QLabel#Subtitle { font-size: 14px; color: #97a3b6; }
            QLabel#SectionTitle { font-size: 17px; font-weight: 700; color: #f0f4fa; }
            QLabel#StatusIdle {
                background: #232833; color: #c8d1e0;
                border: 1px solid #31394a; border-radius: 12px;
                padding: 6px 12px; font-weight: 700;
            }
            QLabel#StatusListening {
                background: #123524; color: #86efac;
                border: 1px solid #1f6f46; border-radius: 12px;
                padding: 6px 12px; font-weight: 700;
            }
            QLabel#StatusThinking {
                background: #382d12; color: #fcd34d;
                border: 1px solid #6f561f; border-radius: 12px;
                padding: 6px 12px; font-weight: 700;
            }
            QLabel#StatusSpeaking {
                background: #1f2940; color: #93c5fd;
                border: 1px solid #314a78; border-radius: 12px;
                padding: 6px 12px; font-weight: 700;
            }
            QPushButton#PrimaryButton {
                background: #2563eb; color: white;
                border: none; border-radius: 12px;
                padding: 12px 18px; font-weight: 700;
            }
            QPushButton#PrimaryButton:hover { background: #2f74ff; }
            QPushButton#PrimaryButton:disabled { background: #273041; color: #7e8aa0; }
            QPushButton#DangerButton {
                background: #7f1d1d; color: #fca5a5;
                border: 1px solid #991b1b; border-radius: 12px;
                padding: 12px 18px; font-weight: 700;
            }
            QPushButton#DangerButton:hover { background: #991b1b; }
            QPushButton#DangerButton:disabled { background: #1e2430; color: #4a5568; border-color: #2d3748; }
            QPushButton#SecondaryButton {
                background: #1e2430; color: #d9e1ee;
                border: 1px solid #30384a; border-radius: 12px;
                padding: 12px 18px; font-weight: 700;
            }
            QPushButton#SecondaryButton:hover { background: #252c3a; }
            QTextEdit#DialogBox, QTextEdit#SystemBox, QTextEdit#InfoBox {
                background: #0c0f14; border: 1px solid #232938;
                border-radius: 12px; padding: 10px;
                color: #d7deea; selection-background-color: #2f74ff;
            }
        """)

    def start_assistant(self):
        if self.bridge.is_running():
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        sys.stdout = StreamRedirector(self._stdout_signaler, mirror=self.original_stdout)
        sys.stderr = StreamRedirector(self._stdout_signaler, mirror=self.original_stderr)
        self.append_system_line("GUI: Jarvis запускается...")
        self.set_status("Запуск...", "StatusThinking")
        self.bridge.start()

    def stop_assistant(self):
        if not self.bridge.is_running():
            return
        self.append_system_line("GUI: Остановка Jarvis...")
        self.stop_btn.setEnabled(False)
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        self.bridge.stop(timeout=6.0)
        from PySide6.QtCore import QMetaObject, Qt as _Qt
        QMetaObject.invokeMethod(self, "_on_stop_done", _Qt.QueuedConnection)

    def _on_stop_done(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.set_status("Остановлен", "StatusIdle")
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.append_system_line("GUI: Jarvis остановлен.")

    # ── Слоты диалога ────────────────────────────────────────────────────────

    def _on_user_text(self, text: str):
        """Вызывается только через Signal — гарантированно в GUI-потоке."""
        if not text:
            return
        self.dialog_buffer.append(("user", text.strip()))

    def _on_assistant_text(self, text: str):
        """Вызывается только через Signal — гарантированно в GUI-потоке."""
        if not text:
            return
        self.dialog_buffer.append(("assistant", text.strip()))

    def on_user_text(self, text: str):
        self._on_user_text(text)

    def on_assistant_text(self, text: str):
        self._on_assistant_text(text)

    def on_system_log(self, text: str):
        if not text:
            return
        self.append_system_line(text)

    def on_state_changed(self, state):
        name = getattr(state, "name", None) or str(state)
        if name == "IDLE":
            self.set_status("Ожидание", "StatusIdle")
        elif name == "LISTENING":
            self.set_status("Слушаю...", "StatusListening")
        elif name == "THINKING":
            self.set_status("Думаю...", "StatusThinking")
        elif name == "SPEAKING":
            self.set_status("Говорю...", "StatusSpeaking")
        elif name == "INTERRUPT_LISTEN":
            self.set_status("Слушаю на фоне...", "StatusSpeaking")
        else:
            self.set_status(str(name), "StatusIdle")

    def on_error(self, text: str):
        self.append_system_line(f"[ERROR] {text}")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    def on_stdout_line(self, text: str):
        if text is None:
            return
        s = str(text).rstrip("\r\n")
        if not s:
            return
        self.append_system_line(s)

    def append_system_line(self, text: str):
        if not text:
            return
        self.system_buffer.append(text if text.endswith("\n") else text + "\n")

    def flush_buffers(self):
        # Системный лог
        if self.system_buffer:
            chunk = "".join(self.system_buffer)
            self.system_buffer.clear()
            self.append_colored_text(self.system_view, chunk, self.pick_system_color(chunk))

        # Диалог — теперь tuple (role, text), красим по роли
        if self.dialog_buffer:
            for role, text in self.dialog_buffer:
                if role == "user":
                    line = f">>> {text}\n"
                    color = "#9be7b0"
                else:
                    line = f"Jarvis: {text}\n"
                    color = "#8fd3ff"
                self.append_colored_text(self.dialog_view, line, color)
            self.dialog_buffer.clear()

    def append_colored_text(self, widget: QTextEdit, text: str, color: str):
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        widget.setTextCursor(cursor)
        widget.ensureCursorVisible()

    def pick_system_color(self, text: str) -> str:
        low = text.lower()
        if "error" in low or "traceback" in low:
            return "#ff8f8f"
        if "router" in low:
            return "#fcd34d"
        if "stt" in low or "tts" in low:
            return "#93c5fd"
        if "wake" in low:
            return "#86efac"
        if "llm" in low or "memory" in low or "web" in low:
            return "#c8d1e0"
        return "#d7deea"

    def set_status(self, text: str, object_name: str = "StatusIdle"):
        self.status_label.setText(str(text))
        self.status_label.setObjectName(object_name)
        style = self.status_label.style()
        style.unpolish(self.status_label)
        style.polish(self.status_label)
        self.status_label.update()

    def clear_logs(self):
        self.dialog_view.clear()
        self.system_view.clear()

    def closeEvent(self, event):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.bridge.is_running():
            self.bridge.stop()
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    window = JarvisWindow()
    window.show()
    sys.exit(app.exec())
