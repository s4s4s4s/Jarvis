
# ─── STT / Whisper ──────────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"
LANG = "ru"

# ─── LLM / Ollama ──────────────────────────────────────────────────────────────────────────
OLLAMA_ROUTER_MODEL  = "qwen2.5:14b-instruct-q4_K_M"
OLLAMA_FAST_MODEL    = "llama3.1:8b-instruct-q5_K_M"
OLLAMA_HEAVY_MODEL   = "qwen2.5:32b-instruct-q4_K_M"
OLLAMA_TIMEOUT       = 60
OLLAMA_HEAVY_TIMEOUT = 180
OLLAMA_RETRIES       = 2
OLLAMA_RETRY_DELAY   = 1.5

# P4: ролевые модели для ProjectAgent. Дефолты — безопасные (уже имеются в Ollama).
# Рекомендуемые (если стянешь):
#   ollama pull qwen2.5-coder:32b-instruct-q4_K_M
#   ollama pull qwen2.5-coder:14b-instruct-q4_K_M
# После этого переопредели значения здесь.
PROJECT_CODER_MODEL    = OLLAMA_HEAVY_MODEL  # потом → "qwen2.5-coder:32b-instruct-q4_K_M"
PROJECT_REVIEWER_MODEL = OLLAMA_HEAVY_MODEL  # потом → "qwen2.5-coder:14b-instruct-q4_K_M"
PROJECT_ARCHITECT_MODEL= OLLAMA_HEAVY_MODEL  # остаётся qwen2.5:32b (рассуждение, не код)
PROJECT_HEALER_MODEL   = OLLAMA_HEAVY_MODEL  # потом → "qwen2.5-coder:14b-instruct-q4_K_M"
PROJECT_INTAKE_MODEL   = OLLAMA_FAST_MODEL   # парсинг запроса — львиная работа на fast-модели
PROJECT_README_MODEL   = OLLAMA_FAST_MODEL
PROJECT_REPORT_MODEL   = OLLAMA_FAST_MODEL

# P9: aider как builder/healer (внешний coding-агент под локальный ollama).
# Переключатель позволяет полностью откатиться на свой путь без изменения кода.
AIDER_ENABLED       = False    # включаем поэтапно после миграции _build_one_file и _heal_loop
# Требуется coder-tier модель в ollama. Скачать:
#   ollama pull qwen2.5-coder:32b-instruct-q4_K_M
# Имя ДОЛЖНО совпадать с выводом `ollama list`.
AIDER_MODEL         = "ollama/qwen2.5-coder:32b-instruct-q4_K_M"
AIDER_TIMEOUT_S     = 300      # жёсткий таймаут на один вызов aider
AIDER_MAX_RETRIES   = 2        # сколько раз ретраим при timeout/сбое subprocess
AIDER_BIN           = "aider"  # путь к aider; если в venv — оставить имя, PATH дорешит
AIDER_API_BASE      = "http://localhost:11434"  # ollama endpoint

# ─── Аудио ────────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_MIC = 16000
CHUNK_SIZE      = 512

SILENCE_MS        = 850
MAX_RECORD_SEC    = 15
MIN_UTTERANCE_SEC = 0.35
PRE_ROLL_SEC      = 2.0

# ─── Wake-детектор ───────────────────────────────────────────────────────────────────────
WAKE_CHECK_SEC              = 1.35
WAKE_MIN_CHECK_INTERVAL_SEC = 0.9
WAKE_FAIL_COOLDOWN_SEC      = 0.8
WAKE_SUCCESS_COOLDOWN_SEC   = 1.5

WAKE_VAD_TRIGGER        = 0.60
WAKE_VAD_HOLD           = 0.40
WAKE_MIN_SPEECH_CHUNKS  = 4
WAKE_MAX_SILENCE_CHUNKS = 8
WAKE_MAX_TEXT_LEN       = 120

# ─── Turn-менеджер ────────────────────────────────────────────────────────────────────────
TURN_VAD_TRIGGER = 0.48
TURN_VAD_HOLD    = 0.30

# ─── Режимы ──────────────────────────────────────────────────────────────────────────────
POST_TTS_GRACE_SEC       = 1.5
POST_INTERRUPT_GRACE_SEC = 0.4
IDLE_TIMEOUT_SEC         = 5.0

# ─── История диалога ────────────────────────────────────────────────────────────────────────
MAX_HISTORY = 6

# ─── Фразы ──────────────────────────────────────────────────────────────────────────────────
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

# ─── TTS ─────────────────────────────────────────────────────────────────────────────────────
EDGE_VOICE = "ru-RU-DmitryNeural"
EDGE_RATE  = "+30%"

# ─── Долгосрочная память ─────────────────────────────────────────────────────────────────────
MEMORY_MAX_FACTS     = 1000          # теперь поддерживаем до 1000 фактов (Level 2 gate)
MEMORY_CONTEXT_FACTS = 10            # топ-N фактов для контекста
MEMORY_SEARCH_FACTS  = 5             # топ-N для memory.search
MEMORY_SIM_THRESHOLD = 0.35          # минимальный cosine score для выдачи

# ─── Embedding-роутер (self-learning) ───────────────────────────────────────────────────
EMBED_MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_THRESHOLD   = 0.82
EMBED_AMBIGUITY   = 0.06

# ─── Векторная память (ChromaDB) ──────────────────────────────────────────────────────────
# sentence-transformers/all-MiniLM-L6-v2 — быстрая, локальная, 90 MB
MEMORY_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_CHROMA_DIR  = "data/chroma_memory"   # относительно ROOT
MEMORY_COLLECTION  = "jarvis_facts"
