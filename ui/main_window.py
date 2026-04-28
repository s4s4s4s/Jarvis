# ui/main_window.py
import sys
import threading

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QLineEdit,
)

from .bridge import JarvisBridge
from voice.state import AssistantState  # noqa: F401


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
        self.bridge.mute_changed.connect(self._on_mute_changed, Qt.QueuedConnection)

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

        # ── Header ──
        header = QFrame(); header.setObjectName("Card")
        hl = QHBoxLayout(header); hl.setContentsMargins(22, 18, 22, 18)
        tb = QVBoxLayout()
        t = QLabel("JARVIS"); t.setObjectName("Title")
        s = QLabel("\u0413\u043e\u043b\u043e\u0441\u043e\u0432\u043e\u0439 \u0430\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442 Jarvis"); s.setObjectName("Subtitle")
        tb.addWidget(t); tb.addWidget(s)

        self.status_label = QLabel("\u041e\u0442\u043a\u043b\u044e\u0447\u0451\u043d")
        self.status_label.setObjectName("StatusIdle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(190); self.status_label.setFixedHeight(38)

        self.settings_btn = QPushButton("\u2699  \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438")
        self.settings_btn.setObjectName("SecondaryButton")
        self.settings_btn.clicked.connect(self._open_settings)

        hl.addLayout(tb)
        hl.addStretch()
        hl.addWidget(self.settings_btn)
        hl.addSpacing(12)
        hl.addWidget(self.status_label)

        # ── Controls ──
        ctrl = QFrame(); ctrl.setObjectName("Card")
        cl = QHBoxLayout(ctrl); cl.setContentsMargins(20, 18, 20, 18); cl.setSpacing(12)

        self.start_btn = QPushButton("\u25b6  \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c Jarvis")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self._start)

        self.stop_btn = QPushButton("\u25a0  \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        self.mute_btn = QPushButton("\U0001f507  \u041e\u0442\u043a\u043b. \u043c\u0438\u043a\u0440\u043e\u0444\u043e\u043d")
        self.mute_btn.setObjectName("MuteButton")
        self.mute_btn.setEnabled(False)   # enabled only when running
        self.mute_btn.clicked.connect(self._toggle_mute)

        self.clear_btn = QPushButton("\U0001f5d1  \u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c")
        self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.clicked.connect(self._clear)

        self.exit_btn = QPushButton("\u2715  \u0412\u044b\u0445\u043e\u0434")
        self.exit_btn.setObjectName("SecondaryButton")
        self.exit_btn.clicked.connect(self.close)

        for b in (self.start_btn, self.stop_btn, self.mute_btn, self.clear_btn, self.exit_btn):
            cl.addWidget(b)
        cl.addStretch()

        # ── Grid ──
        grid = QGridLayout(); grid.setSpacing(16)

        dlg_card = QFrame(); dlg_card.setObjectName("Card")
        dlg_v = QVBoxLayout(dlg_card); dlg_v.setContentsMargins(18, 18, 18, 14); dlg_v.setSpacing(10)
        dlg_lbl = QLabel("\u0414\u0438\u0430\u043b\u043e\u0433"); dlg_lbl.setObjectName("SectionTitle")
        self.dialog_view = QTextEdit(); self.dialog_view.setObjectName("DialogBox"); self.dialog_view.setReadOnly(True)

        input_row = QHBoxLayout(); input_row.setSpacing(8)
        self.text_input = QLineEdit()
        self.text_input.setObjectName("TextInput")
        self.text_input.setPlaceholderText("\u041d\u0430\u043f\u0438\u0448\u0438\u0442\u0435 \u0437\u0430\u043f\u0440\u043e\u0441... (Enter)")
        self.text_input.setFixedHeight(40)
        self.text_input.returnPressed.connect(self._send_text)
        self.send_btn = QPushButton("\u2191 \u041e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c")
        self.send_btn.setObjectName("SendButton")
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self._send_text)
        input_row.addWidget(self.text_input)
        input_row.addWidget(self.send_btn)

        dlg_v.addWidget(dlg_lbl)
        dlg_v.addWidget(self.dialog_view)
        dlg_v.addLayout(input_row)

        def _card(title_text, obj_name):
            f = QFrame(); f.setObjectName("Card")
            v = QVBoxLayout(f); v.setContentsMargins(18,18,18,18); v.setSpacing(10)
            lbl = QLabel(title_text); lbl.setObjectName("SectionTitle")
            box = QTextEdit(); box.setObjectName(obj_name); box.setReadOnly(True)
            v.addWidget(lbl); v.addWidget(box)
            return f, box

        sys_card, self.system_view = _card("\u0421\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0439 \u043b\u043e\u0433", "SystemBox")
        inf_card, self.info_view   = _card("\u0421\u043f\u0440\u0430\u0432\u043a\u0430", "InfoBox")
        self._refresh_info_box()

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
            QPushButton#MuteButton     { background:#7c2d12; color:#fdba74; border:1px solid #9a3412; border-radius:12px; padding:12px 18px; font-weight:700; }
            QPushButton#MuteButton:hover { background:#9a3412; }
            QPushButton#MuteButton:disabled { background:#1e2430; color:#4a5568; border-color:#2d3748; }
            QPushButton#UnmuteButton   { background:#14532d; color:#86efac; border:1px solid #166534; border-radius:12px; padding:12px 18px; font-weight:700; }
            QPushButton#UnmuteButton:hover { background:#166534; }
            QPushButton#SendButton { background:#2563eb; color:white; border:none; border-radius:10px; padding:0 20px; font-weight:700; min-width:110px; }
            QPushButton#SendButton:hover { background:#2f74ff; }
            QPushButton#SendButton:disabled { background:#273041; color:#7e8aa0; }
            QLineEdit#TextInput {
                background:#0c0f14; color:#e8ecf3; border:1px solid #2d3547;
                border-radius:10px; padding:0 14px; font-size:14px;
                selection-background-color:#2563eb;
            }
            QLineEdit#TextInput:focus { border:1px solid #2563eb; }
            QTextEdit#DialogBox,QTextEdit#SystemBox,QTextEdit#InfoBox {
                background:#0c0f14; border:1px solid #232938; border-radius:12px;
                padding:10px; color:#d7deea; selection-background-color:#2f74ff;
            }
        """)

    # ── Mute ──

    def _toggle_mute(self):
        self.bridge.toggle_mute()

    @Slot(bool)
    def _on_mute_changed(self, muted: bool):
        if muted:
            self.mute_btn.setText("\U0001f50a  \u0412\u043a\u043b. \u043c\u0438\u043a\u0440\u043e\u0444\u043e\u043d")
            self.mute_btn.setObjectName("UnmuteButton")
        else:
            self.mute_btn.setText("\U0001f507  \u041e\u0442\u043a\u043b. \u043c\u0438\u043a\u0440\u043e\u0444\u043e\u043d")
            self.mute_btn.setObjectName("MuteButton")
        style = self.mute_btn.style()
        style.unpolish(self.mute_btn)
        style.polish(self.mute_btn)
        self.mute_btn.update()

    # ── Text input ──

    def _send_text(self):
        text = self.text_input.text().strip()
        if not text:
            return
        self.text_input.clear()
        self.send_btn.setEnabled(False)
        self.text_input.setEnabled(False)
        self.bridge.send_text(text)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(300, self._unlock_input)

    def _unlock_input(self):
        self.send_btn.setEnabled(True)
        self.text_input.setEnabled(True)
        self.text_input.setFocus()

    # ── Settings ──

    def _open_settings(self):
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec():
            self._refresh_info_box()
            self._on_system_log("[settings] \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u044b. \u041f\u0435\u0440\u0435\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 Jarvis \u0447\u0442\u043e\u0431\u044b \u043f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u044c \u043c\u0438\u043a\u0440\u043e\u0444\u043e\u043d.")

    def _refresh_info_box(self):
        from core import settings as cfg
        import sounddevice as sd
        mic_idx = cfg.get("mic_device")
        if mic_idx is None:
            mic_str = "\u0421\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0439 \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e"
        else:
            try:
                info = sd.query_devices(mic_idx)
                mic_str = f"[{mic_idx}] {info['name']}"
            except Exception:
                mic_str = f"[{mic_idx}] (\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e)"
        self.info_view.setPlainText(
            "Jarvis \u2014 \u0433\u043e\u043b\u043e\u0441\u043e\u0432\u043e\u0439 \u0430\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442\n"
            f"TTS: Edge TTS (+30%)\n"
            f"STT: Whisper large-v3\n"
            f"LLM: Ollama router+fast+heavy\n"
            f"\u041f\u0430\u043c\u044f\u0442\u044c: \u0434\u043e\u043b\u0433\u043e\u0441\u0440\u043e\u0447\u043d\u0430\u044f\n"
            f"\n"
            f"\U0001f3d9 \u041c\u0438\u043a\u0440\u043e\u0444\u043e\u043d: {mic_str}\n"
            f"\n\u0414\u043b\u044f \u0441\u043c\u0435\u043d\u044b \u043c\u0438\u043a\u0440\u043e\u0444\u043e\u043d\u0430: \u2699 \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u2192 \u0441\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u2192 \u043f\u0435\u0440\u0435\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c."
        )

    # ── Slots ──

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
            "IDLE":             ("\u041e\u0436\u0438\u0434\u0430\u043d\u0438\u0435",           "StatusIdle"),
            "LISTENING":        ("\u0421\u043b\u0443\u0448\u0430\u044e...",         "StatusListening"),
            "THINKING":         ("\u0414\u0443\u043c\u0430\u044e...",          "StatusThinking"),
            "SPEAKING":         ("\u0413\u043e\u0432\u043e\u0440\u044e...",          "StatusSpeaking"),
            "INTERRUPT_LISTEN": ("\u0421\u043b\u0443\u0448\u0430\u044e \u043d\u0430 \u0444\u043e\u043d\u0435...", "StatusSpeaking"),
        }
        txt, obj = mapping.get(name, (str(name), "StatusIdle"))
        self._set_status(txt, obj)
        if name == "IDLE":
            self._unlock_input()

    def _on_error(self, text: str):
        self.system_buffer.append(f"[ERROR] {text}\n")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.mute_btn.setEnabled(False)
        self._unlock_input()

    # ── Start / Stop ──

    def _start(self):
        if self.bridge.is_running():
            return
        try:
            from core import settings as cfg
            import core.config as _cfg
            _cfg.MIC_DEVICE = cfg.get("mic_device")
        except Exception:
            pass
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mute_btn.setEnabled(True)
        self._set_status("\u0417\u0430\u043f\u0443\u0441\u043a...", "StatusThinking")
        self.bridge.start()

    def _stop(self):
        if not self.bridge.is_running():
            return
        self.stop_btn.setEnabled(False)
        self.mute_btn.setEnabled(False)
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        self.bridge.stop(timeout=6.0)
        from PySide6.QtCore import QMetaObject, Qt as _Qt
        QMetaObject.invokeMethod(self, "_after_stop", _Qt.QueuedConnection)

    @Slot()
    def _after_stop(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.mute_btn.setEnabled(False)
        self._set_status("\u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d", "StatusIdle")

    def _clear(self):
        self.dialog_view.clear()
        self.system_view.clear()

    # ── Flush ──

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
        if "mute" in low or "muted" in low:      return "#fdba74"
        if "settings" in low:                     return "#c4b5fd"
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
