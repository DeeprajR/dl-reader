"""Local Ollama provider: the zero-PII-egress option (images and excerpts never leave the host).

Selected with LLM_MODEL=ollama/<model>, e.g. ollama/qwen2.5vl. The server address is Ollama's own
OLLAMA_HOST setting (default http://127.0.0.1:11434). Same prompt, parsing and one-retry rule as
the OpenRouter provider.
"""

import base64
import os

import httpx

from app.schemas import LicenceData
from app.services.providers.base import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    ProviderError,
    extract_with_retry,
)

# Ollama's default address, written as 127.0.0.1: Ollama listens on IPv4 only, and resolving
# "localhost" tries ::1 first, which costs ~2 s per request on Windows before falling back.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Vision models on CPU typically take 30-60 s per image, longer while the model first loads.
OLLAMA_TIMEOUT_S = 180.0


def ollama_url() -> str:
    host = (os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    return host if "://" in host else f"http://{host}"


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)  # seam for tests


async def complete(
    base_url: str, model: str, messages: list[dict], *, json_mode: bool = False
) -> str:
    """One Ollama /api/chat call; failures become ProviderError (retryable where sensible)."""
    payload: dict = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if json_mode:
        payload["format"] = "json"
    unavailable = (
        f"Could not reach Ollama at {base_url}. Start it (`ollama serve`) and pull the model "
        f"(`ollama pull {model}`), or set LLM_MODEL to an OpenRouter model."
    )
    try:
        async with _client(OLLAMA_TIMEOUT_S) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
    except httpx.ConnectError:
        raise ProviderError(unavailable)
    except httpx.TimeoutException:
        raise ProviderError(f"Ollama model {model} did not respond within {OLLAMA_TIMEOUT_S:.0f}s.", retryable=True)
    except httpx.HTTPError as e:
        raise ProviderError(f"Ollama request failed ({type(e).__name__}). {unavailable}", retryable=True)

    if resp.status_code == 404:
        raise ProviderError(
            f"Ollama has no model '{model}'. Run `ollama pull {model}` or change LLM_MODEL."
        )
    if resp.status_code >= 400:
        raise ProviderError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}", retryable=resp.status_code >= 500
        )
    try:
        return resp.json()["message"]["content"] or ""
    except (ValueError, KeyError, TypeError):
        raise ProviderError("Ollama returned an unexpected response.", retryable=True)


class OllamaProvider:
    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = (base_url or ollama_url()).rstrip("/")

    async def extract(self, image_bytes: bytes, media_type: str) -> LicenceData:
        messages: list[dict] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXTRACTION_USER_PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            },
        ]
        return await extract_with_retry(
            lambda msgs: complete(self.base_url, self.model, msgs, json_mode=True),
            messages,
            f"ollama/{self.model}",
        )
