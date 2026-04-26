import time
import ollama

MODEL_ROUTER = "qwen2.5:14b-instruct-q4_K_M"
MODEL_FAST   = "llama3.1:8b-instruct-q5_K_M"
MODEL_HEAVY  = "qwen2.5:32b-instruct-q4_K_M"

_OLLAMA_TIMEOUT = 60      # секунд до таймаута одного запроса
_OLLAMA_RETRIES = 2       # повторов при сбое
_OLLAMA_RETRY_DELAY = 1.5 # секунд между повторами

_client = ollama.Client(timeout=_OLLAMA_TIMEOUT)


def chat(model: str, messages: list[dict], options: dict | None = None) -> str:
    opts = options or {"temperature": 0.2, "num_ctx": 8192}
    last_err = None
    for attempt in range(_OLLAMA_RETRIES + 1):
        try:
            resp = _client.chat(model=model, messages=messages, options=opts)
            return (resp.get("message") or {}).get("content", "").strip()
        except Exception as e:
            last_err = e
            print(f"[ollama] Ошибка (попытка {attempt + 1}): {e}")
            if attempt < _OLLAMA_RETRIES:
                time.sleep(_OLLAMA_RETRY_DELAY)
    print(f"[ollama] Все попытки исчерпаны: {last_err}")
    return ""
