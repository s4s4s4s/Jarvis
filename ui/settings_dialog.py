# ui/settings_dialog.py
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QFrame, QLabel, QVBoxLayout,
)
import sounddevice as sd

from core import settings as cfg


def _get_input_devices() -> list[dict]:
    """Return list of {index, name} for input-capable devices only."""
    devices = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append({"index": i, "name": d["name"]})
    except Exception:
        pass
    return devices


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки Jarvis")
        self.setMinimumWidth(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(20)

        # ── Заголовок ──
        title = QLabel("⚙  Настройки")
        title.setObjectName("SettingsTitle")
        root.addWidget(title)

        # ── Карточка: Микрофон ──
        mic_card = QFrame()
        mic_card.setObjectName("SettingsCard")
        card_layout = QVBoxLayout(mic_card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(12)

        section_lbl = QLabel("🎙  Микрофон")
        section_lbl.setObjectName("SettingsSectionTitle")
        card_layout.addWidget(section_lbl)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight)

        self._mic_combo = QComboBox()
        self._mic_combo.setObjectName("SettingsCombo")
        self._mic_combo.setMinimumHeight(36)

        # Populate: системный дефолт + все input-устройства
        self._devices = _get_input_devices()
        self._mic_combo.addItem("Системный по умолчанию", userData=None)
        for d in self._devices:
            self._mic_combo.addItem(f"[{d['index']}] {d['name']}", userData=d["index"])

        # Restore saved value
        saved = cfg.get("mic_device")
        if saved is None:
            self._mic_combo.setCurrentIndex(0)
        else:
            for i in range(self._mic_combo.count()):
                if self._mic_combo.itemData(i) == saved:
                    self._mic_combo.setCurrentIndex(i)
                    break

        form.addRow("Устройство:", self._mic_combo)

        hint = QLabel(
            "Рекомендуется выбирать WASAPI-версию микрофона "
            "(Windows WASAPI) — она даёт чистый сигнал без MME-обработки."
        )
        hint.setObjectName("SettingsHint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        card_layout.addLayout(form)
        root.addWidget(mic_card)

        # ── Кнопки ──
        btns = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel,
            Qt.Horizontal,
        )
        btns.button(QDialogButtonBox.Save).setText("Сохранить")
        btns.button(QDialogButtonBox.Cancel).setText("Отмена")
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _save(self):
        device_idx = self._mic_combo.currentData()  # None or int
        cfg.set("mic_device", device_idx)

        # Hot-reload config.MIC_DEVICE so next Jarvis start picks it up
        try:
            import core.config as _cfg
            _cfg.MIC_DEVICE = device_idx
        except Exception:
            pass

        self.accept()

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background: #0f1115;
                color: #e8ecf3;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QLabel#SettingsTitle {
                font-size: 20px;
                font-weight: 800;
                color: #f4f7fb;
                padding-bottom: 4px;
            }
            QLabel#SettingsSectionTitle {
                font-size: 15px;
                font-weight: 700;
                color: #c9d5e8;
            }
            QLabel#SettingsHint {
                font-size: 12px;
                color: #7a8799;
            }
            QFrame#SettingsCard {
                background: #171a21;
                border: 1px solid #252a35;
                border-radius: 14px;
            }
            QComboBox#SettingsCombo {
                background: #1e2430;
                color: #d9e1ee;
                border: 1px solid #30384a;
                border-radius: 10px;
                padding: 6px 12px;
                font-size: 13px;
                min-width: 320px;
            }
            QComboBox#SettingsCombo::drop-down {
                border: none;
                width: 28px;
            }
            QComboBox#SettingsCombo QAbstractItemView {
                background: #1a1e28;
                border: 1px solid #2d3547;
                border-radius: 8px;
                color: #d9e1ee;
                selection-background-color: #2563eb;
                outline: none;
            }
            QDialogButtonBox QPushButton {
                background: #1e2430;
                color: #d9e1ee;
                border: 1px solid #30384a;
                border-radius: 10px;
                padding: 9px 22px;
                font-weight: 700;
                min-width: 100px;
            }
            QDialogButtonBox QPushButton[text="Сохранить"] {
                background: #2563eb;
                color: white;
                border: none;
            }
            QDialogButtonBox QPushButton[text="Сохранить"]:hover { background: #2f74ff; }
            QDialogButtonBox QPushButton:hover { background: #252c3a; }
        """)
