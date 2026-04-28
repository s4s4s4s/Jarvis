# core/config.py

# ─── STT / Whisper ──────────────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"
LANG = "ru"

# ─── LLM / Ollama ─────────────────────────────────────────────────────────────────────────────
OLLAMA_ROUTER_MODEL  = "qwen2.5:14b-instruct-q4_K_M"
OLLAMA_FAST_MODEL    = "llama3.1:8b-instruct-q5_K_M"
OLLAMA_HEAVY_MODEL   = "qwen2.5:32b-instruct-q4_K_M"
OLLAMA_TIMEOUT       = 60
OLLAMA_HEAVY_TIMEOUT = 180
OLLAMA_RETRIES       = 2
OLLAMA_RETRY_DELAY   = 1.5

# ─── Аудио ──────────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_MIC = 16000
CHUNK_SIZE      = 512

# Микрофон: fifine Microphone, Windows WASAPI (device 48)
# WASAPI даёт чистый сигнал без MME-обработки — лучше для Silero VAD и Whisper.
# Чтобы сменить устройство: python -c "import sounddevice as sd; print(sd.query_devices())"
# и поставить нужный индекс. None = системный дефолт.
MIC_DEVICE = 48

# FIX: снижен с 1100 → 850 мс: при TURN_VAD_TRIGGER=0.48 / HOLD=0.30
SILENCE_MS        = 850
MAX_RECORD_SEC    = 15
MIN_UTTERANCE_SEC = 0.35
PRE_ROLL_SEC      = 2.0

# ─── Wake-детектор ──────────────────────────────────────────────────────────────────────────
WAKE_CHECK_SEC              = 1.35
WAKE_MIN_CHECK_INTERVAL_SEC = 0.9
WAKE_FAIL_COOLDOWN_SEC      = 0.8
WAKE_SUCCESS_COOLDOWN_SEC   = 1.5

# Снижен с 0.60 → 0.45: fifine на WASAPI даёт более тихий сигнал чем MME,
# при 0.60 wake-детектор не триггерился на нормальную речь.
WAKE_VAD_TRIGGER        = 0.45
WAKE_VAD_HOLD           = 0.30
WAKE_MIN_SPEECH_CHUNKS  = 3
WAKE_MAX_SILENCE_CHUNKS = 8
WAKE_MAX_TEXT_LEN       = 120

# ─── Turn-менеджер ──────────────────────────────────────────────────────────────────────────
TURN_VAD_TRIGGER = 0.40
TURN_VAD_HOLD    = 0.25

# ─── Режимы ────────────────────────────────────────────────────────────────────────────────────
POST_TTS_GRACE_SEC       = 1.5
POST_INTERRUPT_GRACE_SEC = 0.4
IDLE_TIMEOUT_SEC         = 5.0

# ─── История диалога ──────────────────────────────────────────────────────────────────────────
MAX_HISTORY = 6

# ─── Фразы ─────────────────────────────────────────────────────────────────────────────────────
WAKE_PHRASES = [
    "джарвис", "джервис", "джарвисс", "джар",
    "jarvis", "jervis",
    "эй джарвис", "эй джервис", "hey jarvis",
    "дарвис", "зарвис",
    "зараз", "зараза", "заразу", "заразы", "заразе", "заразой", "заразка",
    "жарвис", "харвис",
    "джарвес", "джарбис", "джармис", "джабрис",
    "зарвіс", "джарвіс",
    "дарвис,", "зарвис,", "джарвис,", "джервис,", "зараз,", "зараза,",
]

WAKE_BLOCKLIST = []

IGNORE_PHRASES = [
    "продолжение следует",
    "конец фильма",
    "конец серии",
    "субтитры",
    "перевод",
    "to be continued",
    "the end",
    "credits",
    "opening theme",
    "ending theme",
]

# ─── Пути (assets через ROOT) ─────────────────────────────────────────────────────────────────
EDGE_VOICE = "ru-RU-DmitryNeural"
EDGE_RATE  = "+30%"

# ─── Долгосрочная память ─────────────────────────────────────────────────────────────────
MEMORY_MAX_FACTS     = 500
MEMORY_CONTEXT_FACTS = 20
