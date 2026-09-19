"""OpenRouter (OpenAI-compatible) provider for vision extraction and chat completions."""

import base64
import logging
import os

import openai
from openai import AsyncOpenAI

from app.schemas import LicenceData
from app.services.providers.base import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    JSON_RETRY_PROMPT,
    InvalidJSONError,
    ProviderError,
    parse_licence_json,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 2  # one retry on failure or invalid JSON


def openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ProviderError("OPENROUTER_API_KEY is not set on the server.")
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=LLM_TIMEOUT_S, max_retries=0
    )


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

        last_error: ProviderError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                reply = await complete(self.client, self.model, messages)
            except ProviderError as e:
                if not e.retryable:
                    raise
                logger.warning("Extraction attempt %d with %s failed: %s", attempt, self.model, e)
                last_error = e
                continue
            try:
                return parse_licence_json(reply)
            except InvalidJSONError as e:
                logger.warning("Attempt %d with %s returned invalid JSON: %s", attempt, self.model, e)
                last_error = ProviderError(f"The model {self.model} did not return valid JSON.")
                if reply.strip():  # an empty reply is simply retried as-is
                    messages = messages + [
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": JSON_RETRY_PROMPT},
                    ]
        assert last_error is not None
        raise last_error
