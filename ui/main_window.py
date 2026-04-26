# ui/main_window.py
import sys
import threading

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from .bridge import JarvisBridge

try:
    from voice.state import AssistantState
except Exception:
    AssistantState = None  # type: ignore


class JarvisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jarvis Control Center")
        self.resize(1320, 860)

        self.bridge = JarvisBridge()
        self.bridge.user_text.connect(self._on_user_text, Qt.QueuedConnection)
        self.bridge.assistant_text.connect(self._on_assistant_text, Qt.QueuedConnection)
        self.bridge.system_log.connect(self._on_system_log, Qt.QueuedConnection)
        self.bridge.state_changed.connect(self._on_state_changed, Qt.QueuedConnection)
        self.bridge.error.connect(self._on_error, Qt.QueuedConnection)

        self.dialog_buffer: list[tuple[str, str]] = []
        self.system_buffer: list[str] = []

        self.build_ui()

        from PySide6.QtCore import QTimer
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._flush_buffers)
        self._flush_timer.start(80)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QFrame(); header.setObjectName("Card")
        hl = QHBoxLayout(header); hl.setContentsMargins(22, 18, 22, 18)
        tb = QVBoxLayout()
        t = QLabel("JARVIS"); t.setObjectName("Title")
        s = QLabel("Голосовой ассистент Jarvis"); s.setObjectName("Subtitle")
        tb.addWidget(t); tb.addWidget(s)
        self.status_label = QLabel("Отключён")
        self.status_label.setObjectName("StatusIdle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(190); self.status_label.setFixedHeight(38)
        hl.addLayout(tb); hl.addStretch(); hl.addWidget(self.status_label)

        ctrl = QFrame(); ctrl.setObjectName("Card")
        cl = QHBoxLayout(ctrl); cl.setContentsMargins(20, 18, 20, 18); cl.setSpacing(12)
        self.start_btn = QPushButton("Запустить Jarvis"); self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn  = QPushButton("Остановить");  self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop)
        self.clear_btn = QPushButton("Очистить");     self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.clicked.connect(self._clear)
        self.exit_btn  = QPushButton("Выход");       self.exit_btn.setObjectName("SecondaryButton")
        self.exit_btn.clicked.connect(self.close)
        for b in (self.start_btn, self.stop_btn, self.clear_btn, self.exit_btn):
            cl.addWidget(b)
        cl.addStretch()

        grid = QGridLayout(); grid.setSpacing(16)

        def _card(title_text, obj_name):
            f = QFrame(); f.setObjectName("Card")
            v = QVBoxLayout(f); v.setContentsMargins(18,18,18,18); v.setSpacing(10)
            lbl = QLabel(title_text); lbl.setObjectName("SectionTitle")
            box = QTextEdit(); box.setObjectName(obj_name); box.setReadOnly(True)
            v.addWidget(lbl); v.addWidget(box)
            return f, box

        dlg_card, self.dialog_view = _card("Диалог", "DialogBox")
        sys_card, self.system_view = _card("Системный лог", "SystemBox")
        inf_card, self.info_view   = _card("Справка", "InfoBox")
        self.info_view.setPlainText(
            "Jarvis — голосовой ассистент\n"
            "TTS: Edge TTS (streaming, +20%)\nSTT: Whisper large-v3\n"
            "LLM: Ollama router+fast+heavy\nПамять: долгосрочная"
        )

        grid.addWidget(dlg_card, 0, 0, 2, 2)
        grid.addWidget(sys_card, 0, 2, 1, 1)
        grid.addWidget(inf_card, 1, 2, 1, 1)
        grid.setColumnStretch(0, 3); grid.setColumnStretch(1, 2); grid.setColumnStretch(2, 2)
        grid.setRowStretch(0, 3);    grid.setRowStretch(1, 2)

        layout.addWidget(header); layout.addWidget(ctrl); layout.addLayout(grid)
        self.setStyleSheet("""
            QWidget { background:#0f1115; color:#e8ecf3; font-family:"Segoe UI",sans-serif; font-size:14px; }
            QMainWindow { background:#0f1115; }
            QFrame#Card { background:#171a21; border:1px solid #252a35; border-radius:16px; }
            QLabel#Title { font-size:30px; font-weight:800; color:#f4f7fb; letter-spacing:1px; }
            QLabel#Subtitle { font-size:14px; color:#97a3b6; }
            QLabel#SectionTitle { font-size:17px; font-weight:700; color:#f0f4fa; }
            QLabel#StatusIdle    { background:#232833; color:#c8d1e0; border:1px solid #31394a; border-radius:12px; padding:6px 12px; font-weight:700; }
            QLabel#StatusListening{ background:#123524; color:#86efac; border:1px solid #1f6f46; border-radius:12px; padding:6px 12px; font-weight:700; }
            QLabel#StatusThinking { background:#382d12; color:#fcd34d; border:1px solid #6f561f; border-radius:12px; padding:6px 12px; font-weight:700; }
            QLabel#StatusSpeaking { background:#1f2940; color:#93c5fd; border:1px solid #314a78; border-radius:12px; padding:6px 12px; font-weight:700; }
            QPushButton#PrimaryButton  { background:#2563eb; color:white; border:none; border-radius:12px; padding:12px 18px; font-weight:700; }
            QPushButton#PrimaryButton:hover { background:#2f74ff; }
            QPushButton#PrimaryButton:disabled { background:#273041; color:#7e8aa0; }
            QPushButton#DangerButton   { background:#7f1d1d; color:#fca5a5; border:1px solid #991b1b; border-radius:12px; padding:12px 18px; font-weight:700; }
            QPushButton#DangerButton:hover { background:#991b1b; }
            QPushButton#DangerButton:disabled { background:#1e2430; color:#4a5568; border-color:#2d3748; }
            QPushButton#SecondaryButton{ background:#1e2430; color:#d9e1ee; border:1px solid #30384a; border-radius:12px; padding:12px 18px; font-weight:700; }
            QPushButton#SecondaryButton:hover { background:#252c3a; }
            QTextEdit#DialogBox,QTextEdit#SystemBox,QTextEdit#InfoBox {
                background:#0c0f14; border:1px solid #232938; border-radius:12px;
                padding:10px; color:#d7deea; selection-background-color:#2f74ff;
            }
        """)

    # ---- Слоты ----

    def _on_user_text(self, text: str):
        if text:
            self.dialog_buffer.append(("user", text.strip()))

    def _on_assistant_text(self, text: str):
        if text:
            self.dialog_buffer.append(("assistant", text.strip()))

    def _on_system_log(self, text: str):
        if text:
            self.system_buffer.append(text if text.endswith("\n") else text + "\n")

    def _on_state_changed(self, state):
        name = getattr(state, "name", None) or str(state)
        mapping = {
            "IDLE":             ("Ожидание",           "StatusIdle"),
            "LISTENING":        ("Слушаю...",         "StatusListening"),
            "THINKING":         ("Думаю...",          "StatusThinking"),
            "SPEAKING":         ("Говорю...",          "StatusSpeaking"),
            "INTERRUPT_LISTEN": ("Слушаю на фоне...", "StatusSpeaking"),
        }
        txt, obj = mapping.get(name, (str(name), "StatusIdle"))
        self._set_status(txt, obj)

    def _on_error(self, text: str):
        self.system_buffer.append(f"[ERROR] {text}\n")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    # ---- Запуск / стоп ----

    def _start(self):
        if self.bridge.is_running():
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("Запуск...", "StatusThinking")
        self.bridge.start()

    def _stop(self):
        if not self.bridge.is_running():
            return
        self.stop_btn.setEnabled(False)
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        self.bridge.stop(timeout=6.0)
        from PySide6.QtCore import QMetaObject, Qt as _Qt
        # FIX: _after_stop помечен @Slot() — без этого PySide6 invokeMethod
        # по имени строки не находит метод и кнопка "Запустить" зависает
        QMetaObject.invokeMethod(self, "_after_stop", _Qt.QueuedConnection)

    # FIX: декоратор @Slot() обязателен для QMetaObject.invokeMethod по имени
    @Slot()
    def _after_stop(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("Остановлен", "StatusIdle")

    def _clear(self):
        self.dialog_view.clear()
        self.system_view.clear()

    # ---- Flush ----

    def _flush_buffers(self):
        if self.system_buffer:
            chunk = "".join(self.system_buffer)
            self.system_buffer.clear()
            self._append(self.system_view, chunk, self._sys_color(chunk))

        for role, text in self.dialog_buffer:
            if role == "user":
                self._append(self.dialog_view, f">>> {text}\n", "#9be7b0")
            else:
                self._append(self.dialog_view, f"Jarvis: {text}\n", "#8fd3ff")
        self.dialog_buffer.clear()

    def _append(self, widget: QTextEdit, text: str, color: str):
        cur = widget.textCursor()
        cur.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cur.setCharFormat(fmt)
        cur.insertText(text)
        widget.setTextCursor(cur)
        widget.ensureCursorVisible()

    def _sys_color(self, text: str) -> str:
        low = text.lower()
        if "error" in low or "traceback" in low: return "#ff8f8f"
        if "router" in low:                       return "#fcd34d"
        if "stt" in low or "tts" in low:         return "#93c5fd"
        if "wake" in low:                         return "#86efac"
        return "#d7deea"

    def _set_status(self, text: str, obj: str = "StatusIdle"):
        self.status_label.setText(text)
        self.status_label.setObjectName(obj)
        style = self.status_label.style()
        style.unpolish(self.status_label)
        style.polish(self.status_label)
        self.status_label.update()

    def closeEvent(self, event):
        if self.bridge.is_running():
            self.bridge.stop()
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    window = JarvisWindow()
    window.show()
    sys.exit(app.exec())
