# voice/state.py
"""
Состояния ассистента — единый источник правды.
Импортируется из voice/assistant.py, ui/bridge.py, ui/main_window.py.
"""
from enum import Enum, auto


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPT_LISTEN = auto()
