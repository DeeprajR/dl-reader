"""Step 10 (Phase 2, optional stretch) - Ollama provider, the zero-PII-egress option.

Interface:
  get_provider("ollama/<model>") -> OllamaProvider with .model == "<model>"
  OllamaProvider(model, base_url="http://127.0.0.1:11434")
  an unreachable server raises ProviderError advising the LLM_MODEL change (never a crash)
  same prompt, JSON parsing and one-retry rule as the OpenRouter provider
"""

import asyncio
import base64
import json
import logging

import httpx
import pytest
from conftest import REPO_DIR, image_bytes, make_licence

from app.services.providers.base import (
    EXTRACTION_SYSTEM_PROMPT,
    JSON_RETRY_PROMPT,
    ProviderError,
    get_provider,
)

pytestmark = [pytest.mark.phase2, pytest.mark.step10]


def licence_reply() -> str:
    data = make_licence().model_dump()
    return json.dumps({k: ({"value": v["value"], "source_text": v["source_text"]} if k != "other_fields" else {})
                       for k, v in data.items()})


def fake_ollama(monkeypatch, *replies, status=200):
    """Serve /api/chat from `replies` in order via httpx.MockTransport; returns the request log."""
    from app.services.providers import ollama

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "model not found"})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": replies[len(requests) - 1]}})

    monkeypatch.setattr(ollama, "_client", lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return requests


def extract(provider):
    return asyncio.run(provider.extract(image_bytes((40, 30)), "image/png"))


def test_factory_routes_ollama_models():
    provider = get_provider("ollama/qwen2.5vl")
    assert type(provider).__name__ == "OllamaProvider"
    assert provider.model == "qwen2.5vl"


def test_unreachable_ollama_is_a_clean_error():
    from app.services.providers.ollama import OllamaProvider

    provider = OllamaProvider("qwen2.5vl", base_url="http://127.0.0.1:9")  # nothing listens here
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.extract(image_bytes((40, 30)), "image/png"))
    assert "LLM_MODEL" in str(info.value)


def test_request_uses_the_same_prompt_with_the_image(monkeypatch):
    requests = fake_ollama(monkeypatch, licence_reply())
    data = extract(get_provider("ollama/qwen2.5vl"))
    assert data.full_name.value == "JOHN DOE"

    sent = requests[0]
    assert sent["model"] == "qwen2.5vl" and sent["stream"] is False and sent["format"] == "json"
    system, user = sent["messages"]
    assert system == {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}
    assert base64.b64decode(user["images"][0]) == image_bytes((40, 30))


def test_invalid_json_is_retried_once_with_the_retry_prompt(monkeypatch):
    requests = fake_ollama(monkeypatch, "not json", licence_reply())
    assert extract(get_provider("ollama/qwen2.5vl")).licence_number.value == "MH12 20190001234"
    assert len(requests) == 2
    assert requests[1]["messages"][-1] == {"role": "user", "content": JSON_RETRY_PROMPT}


def test_missing_model_is_a_clean_error(monkeypatch):
    requests = fake_ollama(monkeypatch, status=404)
    with pytest.raises(ProviderError, match="ollama pull qwen2.5vl"):
        extract(get_provider("ollama/qwen2.5vl"))
    assert len(requests) == 1  # not retried


def test_local_chat_model_keeps_chat_on_ollama(client, extracted, monkeypatch):
    from app.services.providers import ollama

    calls = []

    async def local_complete(base_url, model, messages, *, json_mode=False):
        calls.append(model)
        return "The licence number is MH12 20190001234."

    monkeypatch.setattr(ollama, "complete", local_complete)
    monkeypatch.setenv("LLM_CHAT_MODEL", "ollama/llama3.2")
    response = client.post(f"/api/documents/{extracted}/chat", json={"question": "What is the licence number?"})
    assert response.status_code == 200 and calls == ["llama3.2"]  # the OpenRouter path would raise


def test_all_local_setup_starts_without_openrouter_key(monkeypatch, caplog):
    from app import main

    monkeypatch.setenv("LLM_MODEL", "ollama/qwen2.5vl")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    with caplog.at_level(logging.WARNING):
        main.startup_checks()  # no RuntimeError: nothing is sent to OpenRouter
    assert "Ollama is not reachable" in caplog.text

    monkeypatch.setenv("LLM_CHAT_MODEL", "vendor/cloud-chat")  # cloud chat -> key required again
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        main.startup_checks()


def test_readme_documents_the_local_option():
    readme = (REPO_DIR / "README.md").read_text(encoding="utf-8").lower()
    assert "ollama" in readme and "pii" in readme and "ollama_host" in readme
