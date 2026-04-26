# voice/state.py

from enum import Enum, auto


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPT_LISTEN = auto()
