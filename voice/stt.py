from __future__ import annotations


def transcribe(model, audio_path: str):
    return model.transcribe(
        audio_path,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
