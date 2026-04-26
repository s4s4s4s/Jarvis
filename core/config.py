# core/config.py

# ─── STT / Whisper ────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"
LANG = "ru"

# ─── LLM / Ollama ─────────────────────────────────────────────────────────────
OLLAMA_ROUTER_MODEL  = "qwen2.5:14b-instruct-q4_K_M"
OLLAMA_FAST_MODEL    = "llama3.1:8b-instruct-q5_K_M"
OLLAMA_HEAVY_MODEL   = "qwen2.5:32b-instruct-q4_K_M"
OLLAMA_TIMEOUT       = 60      # seconds per request
OLLAMA_RETRIES       = 2
OLLAMA_RETRY_DELAY   = 1.5

# ─── Аудио ────────────────────────────────────────────────────────────────────
SAMPLE_RATE_MIC = 16000
CHUNK_SIZE      = 512

SILENCE_MS        = 1100
MAX_RECORD_SEC    = 15
MIN_UTTERANCE_SEC = 0.35
PRE_ROLL_SEC      = 2.0

# ─── Wake-детектор ────────────────────────────────────────────────────────────
WAKE_CHECK_SEC              = 1.35
WAKE_MIN_CHECK_INTERVAL_SEC = 0.9
WAKE_FAIL_COOLDOWN_SEC      = 0.8
WAKE_SUCCESS_COOLDOWN_SEC   = 1.5

WAKE_VAD_TRIGGER        = 0.60
WAKE_VAD_HOLD           = 0.40
WAKE_MIN_SPEECH_CHUNKS  = 4
WAKE_MAX_SILENCE_CHUNKS = 8
WAKE_MAX_TEXT_LEN       = 120

# ─── Turn-менеджер ────────────────────────────────────────────────────────────
TURN_VAD_TRIGGER = 0.48
TURN_VAD_HOLD    = 0.30

# ─── Режимы ───────────────────────────────────────────────────────────────────
POST_TTS_GRACE_SEC       = 2.5
POST_INTERRUPT_GRACE_SEC = 0.4
IDLE_TIMEOUT_SEC         = 5.0

# ─── История диалога ──────────────────────────────────────────────────────────
MAX_HISTORY = 6  # ходов (каждый ход = user + assistant)

# ─── Фразы ────────────────────────────────────────────────────────────────────
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

# ─── TTS ──────────────────────────────────────────────────────────────────────
REFERENCE_WAV = "C:/jarvis/assets/reference.wav"
EDGE_VOICE    = "ru-RU-DmitryNeural"

# ─── Звуки ────────────────────────────────────────────────────────────────────
ACTIVATE_SOUND_PATH   = "C:/jarvis/assets/activate.wav"
DEACTIVATE_SOUND_PATH = "C:/jarvis/assets/deactivate.wav"

# ─── Долгосрочная память ──────────────────────────────────────────────────────
# Paths live in core/paths.py — do not duplicate here.
MEMORY_MAX_FACTS     = 500
MEMORY_CONTEXT_FACTS = 20
