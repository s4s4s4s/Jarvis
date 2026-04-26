from __future__ import annotations

import threading

import torch

from core.config import WHISPER_MODEL_SIZE, SAMPLE_RATE_MIC, LANG

_lock = threading.Lock()
_whisper_model = None
_vad_model = None


def _ensure_models():
    global _whisper_model, _vad_model
    if _whisper_model is not None:
        return
    with _lock:
        if _whisper_model is not None:
            return
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"

        print(f"[STT] Загружаю Whisper {WHISPER_MODEL_SIZE} ({device}/{compute})...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute)
        print("[STT] Загружаю Silero VAD...")
        _vad_model = load_silero_vad()
        print("[STT] Модели загружены.")


def transcribe(audio, log: bool = True) -> str:
    _ensure_models()
    if log:
        print("[STT] Распознаю...")
    audio_len_sec = len(audio) / SAMPLE_RATE_MIC
    # FIX: порог снижен с 3.0 → 1.5 сек: короткие команды ("Джарвис, который час?"
    # ≈ 2 сек) получают beam=3 вместо beam=1, что даёт заметно лучшую точность
    # на русском языке с Whisper large-v3
    beam = 1 if audio_len_sec < 1.5 else 3
    segments, _ = _whisper_model.transcribe(
        audio,
        language=LANG,
        beam_size=beam,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    text = "".join(seg.text for seg in segments).strip()
    if log:
        print(f"[STT] Ты: {text}")
    return text


def vad_prob(chunk) -> float:
    _ensure_models()
    tensor = torch.from_numpy(chunk.astype("float32"))
    return _vad_model(tensor, SAMPLE_RATE_MIC).item()
