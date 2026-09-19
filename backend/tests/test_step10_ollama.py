"""Step 10 (Phase 2, optional stretch) - Ollama provider. Pending until 10 is done.

Interface defined here:
  get_provider("ollama/<model>") -> OllamaProvider with .model == "<model>"
  OllamaProvider(model, base_url="http://localhost:11434")
  an unreachable server raises ProviderError advising the LLM_MODEL change (never a crash)
"""

import asyncio

import pytest
from conftest import REPO_DIR, image_bytes

from app.services.providers.base import ProviderError, get_provider

pytestmark = [pytest.mark.phase2, pytest.mark.step10]


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


def test_readme_documents_the_local_option():
    readme = (REPO_DIR / "README.md").read_text(encoding="utf-8").lower()
    assert "ollama" in readme and "pii" in readme
