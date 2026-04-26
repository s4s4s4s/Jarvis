# voice/sounds.py
from pathlib import Path
import pygame

# FIX (audit 3): пути ассетов переехали в core/paths.py — единый источник
# правды, реагирует на JARVIS_ROOT.
from core.paths import ACTIVATE_SOUND_PATH, DEACTIVATE_SOUND_PATH

_MIXER_FREQ = 44100
_MIXER_SIZE = -16
_MIXER_CHANNELS = 2
_MIXER_BUFFER = 512

_activate_sound = None
_deactivate_sound = None


def _ensure_mixer():
    if not pygame.mixer.get_init():
        pygame.mixer.init(
            frequency=_MIXER_FREQ,
            size=_MIXER_SIZE,
            channels=_MIXER_CHANNELS,
            buffer=_MIXER_BUFFER,
        )


def _load_sound(path: str) -> pygame.mixer.Sound:
    _ensure_mixer()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл звука не найден: {p}")
    return pygame.mixer.Sound(str(p))


def play_activate():
    global _activate_sound
    if _activate_sound is None:
        _activate_sound = _load_sound(ACTIVATE_SOUND_PATH)
    _activate_sound.play()


def play_deactivate():
    global _deactivate_sound
    if _deactivate_sound is None:
        _deactivate_sound = _load_sound(DEACTIVATE_SOUND_PATH)
    _deactivate_sound.play()