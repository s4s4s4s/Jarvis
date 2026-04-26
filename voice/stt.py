import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from core.config import WHISPER_MODEL_SIZE, SAMPLE_RATE_MIC, LANG

device = "cuda" if torch.cuda.is_available() else "cpu"
compute = "float16" if device == "cuda" else "int8"

print("Загружаю Whisper...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=device,
    compute_type=compute,
)

print("Загружаю Silero VAD...")
vad_model = load_silero_vad()


def transcribe(audio, log=True):
    if log:
        print("Распознаю...")
    audio_len_sec = len(audio) / SAMPLE_RATE_MIC
    beam = 1 if audio_len_sec < 3.0 else 3
    segments, _ = whisper_model.transcribe(
        audio,
        language=LANG,
        beam_size=beam,
        temperature=0.0,
        vad_filter=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    if log:
        print(f"Ты: {text}")
    return text


def vad_prob(chunk):
    tensor = torch.from_numpy(chunk.astype("float32"))
    return vad_model(tensor, SAMPLE_RATE_MIC).item()
