"""OpenRouter (OpenAI-compatible) provider for vision extraction and chat completions."""

import base64
import functools
import os

import openai
from openai import AsyncOpenAI

from app.schemas import LicenceData
from app.services.providers.base import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    ProviderError,
    extract_with_retry,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT_S = 60.0


@functools.lru_cache(maxsize=2)
def _client_for(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=LLM_TIMEOUT_S, max_retries=0)


def openrouter_client() -> AsyncOpenAI:
    """Shared client (one connection pool) for extraction and chat."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ProviderError("OPENROUTER_API_KEY is not set on the server.")
    return _client_for(api_key)


async def complete(client: AsyncOpenAI, model: str, messages: list[dict]) -> str:
    """One chat completion; OpenAI SDK errors become ProviderError (retryable where sensible)."""
    try:
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=8192
        )
    except openai.AuthenticationError:
        raise ProviderError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY.")
    except openai.APITimeoutError:
        raise ProviderError(f"The model {model} did not respond within {LLM_TIMEOUT_S:.0f}s.", retryable=True)
    except openai.APIConnectionError:
        raise ProviderError("Could not reach OpenRouter. Check the network connection.", retryable=True)
    except openai.APIStatusError as e:
        retryable = e.status_code == 429 or e.status_code >= 500
        raise ProviderError(
            f"OpenRouter returned HTTP {e.status_code} for model {model}: {e.message}", retryable=retryable
        )
    if not resp.choices:
        detail = getattr(resp, "error", None) or "no choices returned"
        raise ProviderError(f"The model {model} returned no answer ({detail}).", retryable=True)
    return resp.choices[0].message.content or ""


class OpenRouterProvider:
    """Extraction with any OpenRouter vision model (the image is sent as a data URL)."""

    def __init__(self, model: str):
        self.model = model
        self.client = openrouter_client()

    async def extract(self, image_bytes: bytes, media_type: str) -> LicenceData:
        data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        messages: list[dict] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return await extract_with_retry(
            lambda msgs: complete(self.client, self.model, msgs), messages, self.model
        )
