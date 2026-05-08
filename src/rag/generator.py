import os
import sys
from typing import Generator

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.config.settings import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    MAX_TOKENS, TEMPERATURE
)
import requests


# ── Groq — standard (returns full string) ─────────────
def _call_groq(prompt: str) -> str:
    from groq import Groq
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content.strip()


# ── Groq — streaming (yields chunks) ──────────────────
def _stream_groq(prompt: str) -> Generator[str, None, None]:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Ollama — standard ──────────────────────────────────
def _call_ollama(prompt: str) -> str:
    payload  = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload, timeout=60
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ── Ollama — streaming ─────────────────────────────────
def _stream_ollama(prompt: str) -> Generator[str, None, None]:
    payload  = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload, stream=True, timeout=60
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            try:
                import json
                obj   = json.loads(data)
                token = obj.get("response", "")
                if token:
                    yield token
                if obj.get("done"):
                    break
            except Exception:
                continue


# ── Public API ─────────────────────────────────────────
def generate_answer(prompt: str) -> str:
    """Standard (blocking) generation — returns full answer string."""
    provider = LLM_PROVIDER
    if provider == "groq" and not GROQ_API_KEY:
        print("⚠️  No Groq key — falling back to Ollama.")
        provider = "ollama"

    if provider == "groq":
        return _call_groq(prompt)
    elif provider == "ollama":
        return _call_ollama(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def stream_answer(prompt: str) -> Generator[str, None, None]:
    """
    Streaming generation — yields tokens one by one.
    Use with st.write_stream() in Streamlit.
    """
    provider = LLM_PROVIDER
    if provider == "groq" and not GROQ_API_KEY:
        provider = "ollama"

    if provider == "groq":
        yield from _stream_groq(prompt)
    elif provider == "ollama":
        yield from _stream_ollama(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")