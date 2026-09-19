"""Step 5 - startup checks, error handling, .env.example and the security baseline."""

import logging
import shutil
import subprocess

import pytest
from conftest import REPO_DIR, TEST_API_KEY

from app import main
from app.services import ocr, storage

pytestmark = [pytest.mark.phase1, pytest.mark.step5]


def test_startup_fails_fast_without_api_key(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        main.startup_checks()

    storage.init(tmp_path / "data")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        with TestClient(main.app):
            pass


@pytest.mark.parametrize(
    ("host", "shown"),
    [("0.0.0.0", "http://localhost:7860"), ("127.0.0.1", "http://127.0.0.1:7860")],
)
def test_startup_message_shows_an_openable_address(caplog, host, shown):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        logging.getLogger("uvicorn.error").info("Uvicorn running on %s://%s:%d", "http", host, 7860)
    assert shown in caplog.text
    assert "0.0.0.0" not in caplog.text


def test_startup_only_warns_when_ocr_tools_are_missing(monkeypatch, caplog):
    def missing():
        raise FileNotFoundError("tesseract")

    monkeypatch.setattr(ocr, "tesseract_version", missing)
    monkeypatch.setenv("POPPLER_PATH", "/definitely/not/here")
    with caplog.at_level(logging.WARNING):
        main.startup_checks()  # does not raise
    assert "Tesseract was not found" in caplog.text
    assert "poppler" in caplog.text


def test_unhandled_errors_return_generic_json_without_details(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    def explode():
        raise RuntimeError("secret internal detail at C:\\path\\file.py")

    storage.init(tmp_path / "data")
    monkeypatch.setattr(storage, "list_documents", explode)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.get("/api/documents")
    assert response.status_code == 500
    assert response.json() == {"error": "An unexpected server error occurred."}
    assert "secret" not in response.text and "Traceback" not in response.text


def test_validation_errors_are_readable_json(client, uploaded):
    response = client.post(f"/api/documents/{uploaded}/chat", data="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"].startswith("Invalid request")


def test_api_key_is_never_logged(client, uploaded, fake_provider, fake_ocr, monkeypatch, caplog):
    from app.services import rag

    async def reply(client_, model, messages):
        return "The licence number is MH12 20190001234."

    monkeypatch.setattr(rag, "complete", reply)
    with caplog.at_level(logging.DEBUG):
        main.startup_checks()
        client.post(f"/api/documents/{uploaded}/extract")
        client.post(f"/api/documents/{uploaded}/chat", json={"question": "What is the licence number?"})
    assert caplog.records, "expected some log output"
    assert TEST_API_KEY not in caplog.text


def test_env_example_lists_every_setting_with_spec_defaults():
    lines = (REPO_DIR / "backend" / ".env.example").read_text(encoding="utf-8").splitlines()
    settings = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
    assert settings == {
        "OPENROUTER_API_KEY": "",
        "LLM_MODEL": "google/gemini-3.8-flash",
        "LLM_MODEL_ALT": "anthropic/claude-sonnet-5",
        "LLM_CHAT_MODEL": "",
        "MAX_UPLOAD_MB": "10",
        "TESSERACT_CMD": "",
        "POPPLER_PATH": "",
    }


def test_gitignore_keeps_secrets_pii_and_build_output_out():
    entries = {line.strip() for line in (REPO_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()}
    for required in (".env", "data/", "chroma/", "samples/", "node_modules/", "__pycache__/", "dist/"):
        assert required in entries, required


def test_no_documents_or_secrets_are_tracked_by_git():
    if not shutil.which("git") or not (REPO_DIR / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_DIR, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    forbidden_ext = (".png", ".jpg", ".jpeg", ".pdf", ".db", ".sqlite")
    offenders = [
        f for f in tracked
        if f.lower().endswith(forbidden_ext) or f.split("/")[0] in {"samples", "data", "chroma"}
        or f.endswith(".env") or "/data/" in f or "/chroma/" in f
    ]
    assert offenders == []
