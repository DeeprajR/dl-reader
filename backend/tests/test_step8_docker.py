"""Step 8 (Phase 2) - Dockerfile structure.

These static checks define the container contract; the real verification is
`docker build -t licence-reader . && docker run -p 7860:7860 --env-file .env licence-reader`
plus an extraction against the running container (too slow for the unit-test budget).
"""

import re

import pytest
from conftest import REPO_DIR

pytestmark = [pytest.mark.phase2, pytest.mark.step8]


def dockerfile() -> str:
    return (REPO_DIR / "Dockerfile").read_text(encoding="utf-8")


def test_multi_stage_node_build_then_python_runtime():
    stages = re.findall(r"^FROM\s+(\S+)", dockerfile(), flags=re.MULTILINE | re.IGNORECASE)
    assert len(stages) == 2
    assert stages[0].startswith("node:20") and stages[0].endswith("-slim")
    assert stages[1] == "python:3.11-slim"


def test_runtime_installs_ocr_tools_and_builds_frontend():
    text = dockerfile()
    assert re.search(r"apt-get install[^\n]*tesseract-ocr", text) and "poppler-utils" in text
    assert "vite build" in text or "npm run build" in text
    assert "requirements.txt" in text


def test_serves_on_port_7860_configurable_via_port():
    text = dockerfile()
    assert "7860" in text
    assert "uvicorn" in text and "app.main:app" in text
    assert re.search(r"\$\{?PORT", text), "port must be configurable via the PORT env var"


def test_dockerignore_keeps_secrets_and_documents_out_of_the_image():
    entries = {line.strip().rstrip("/") for line in (REPO_DIR / ".dockerignore").read_text().splitlines()}
    for required in (".env", "samples", "data", "chroma", "node_modules"):
        assert any(e.lstrip("*/").startswith(required) for e in entries), required
