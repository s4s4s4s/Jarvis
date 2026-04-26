# C:\jarvis\brain\client.py
import ollama

MODEL_ROUTER = "qwen2.5:14b-instruct-q4_K_M"
MODEL_FAST   = "llama3.1:8b-instruct-q5_K_M"
MODEL_HEAVY  = "qwen2.5:32b-instruct-q4_K_M"

_client = ollama.Client()


def chat(model: str, messages: list[dict], options: dict | None = None) -> str:
    resp = _client.chat(
        model=model,
        messages=messages,
        options=options or {"temperature": 0.2, "num_ctx": 8192},
    )
    return (resp.get("message") or {}).get("content", "").strip()
# === end of file: brain/client.py ===