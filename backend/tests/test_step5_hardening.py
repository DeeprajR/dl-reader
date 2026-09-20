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
    """Without an API key the server refuses to start, with a message that says how to fix it."""
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
    """The startup line shows localhost, never 0.0.0.0, and other addresses are left alone."""
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        logging.getLogger("uvicorn.error").info("Uvicorn running on %s://%s:%d", "http", host, 7860)
    assert shown in caplog.text
    assert "0.0.0.0" not in caplog.text


def test_startup_only_warns_when_ocr_tools_are_missing(monkeypatch, caplog):
    """Missing Tesseract or poppler is a warning, not a crash: the app still starts."""
    def missing():
        raise FileNotFoundError("tesseract")

    monkeypatch.setattr(ocr, "tesseract_version", missing)
    monkeypatch.setenv("POPPLER_PATH", "/definitely/not/here")
    with caplog.at_level(logging.WARNING):
        main.startup_checks()  # does not raise
    assert "Tesseract was not found" in caplog.text
    assert "poppler" in caplog.text


def test_database_from_an_older_version_is_cleaned_up(tmp_path):
    """Columns that older versions stored but never read are dropped, and the documents survive."""
    import sqlite3

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    conn = sqlite3.connect(data_dir / "app.db")
    conn.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, filename_label TEXT NOT NULL, original_name TEXT NOT NULL,"
        " image_name TEXT NOT NULL, media_type TEXT NOT NULL, page_number INTEGER NOT NULL, page_offsets TEXT,"
        " width INTEGER NOT NULL, height INTEGER NOT NULL, uploaded_at TEXT NOT NULL, extraction TEXT, ocr_words TEXT)"
    )
    conn.execute(
        "INSERT INTO documents VALUES ('d1', 'old.png', 'd1.png', 'd1.png', 'image/png', 1, '[0]', 8, 6,"
        " '2026-09-19T00:00:00+00:00', NULL, NULL)"
    )
    conn.commit()
    conn.close()

    storage.init(data_dir)
    doc = storage.get_document("d1")
    assert doc["filename_label"] == "old.png" and doc["width"] == 8
    assert "page_number" not in doc and "page_offsets" not in doc


def test_dates_saved_by_an_older_version_become_day_first(client, extracted):
    """A document saved with year-first dates is rewritten as DD-MM-YYYY at start-up; printed text is left alone."""
    import json
    import sqlite3

    db = storage._data_dir / "app.db"
    conn = sqlite3.connect(db)
    result = json.loads(conn.execute("SELECT extraction FROM documents").fetchone()[0])
    result["data"]["date_of_expiry"]["value"] = "2034-06-15"
    result["data"]["other_fields"]["lmv_valid_till"] = {**result["data"]["date_of_expiry"], "source_text": "LMV 2019-06-16"}
    result["warnings"] = ["Date of issue (2099-01-01) is in the future."]
    conn.execute("UPDATE documents SET extraction = ?", (json.dumps(result),))
    conn.commit()
    conn.close()

    storage.init()
    stored = client.get(f"/api/documents/{extracted}/extract").json()
    assert stored["data"]["date_of_expiry"]["value"] == "15-06-2034"
    assert stored["data"]["other_fields"]["lmv_valid_till"]["value"] == "15-06-2034"
    assert stored["data"]["other_fields"]["lmv_valid_till"]["source_text"] == "LMV 2019-06-16"
    assert stored["warnings"] == ["Date of issue (01-01-2099) is in the future."]


def test_no_cors_headers_are_sent(client):
    """The frontend is always served from the API's own address, so no other site is allowed in."""
    response = client.get("/api/documents", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_unhandled_errors_return_generic_json_without_details(monkeypatch, tmp_path):
    """An unexpected error gives a generic 500 message: no file paths and no stack trace leak out."""
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
    """A malformed request body gives a readable 422, not FastAPI's raw error list."""
    response = client.post(f"/api/documents/{uploaded}/chat", data="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"].startswith("Invalid request")


def test_api_key_is_never_logged(client, uploaded, fake_provider, fake_ocr, monkeypatch, caplog):
    """Security: a full run (startup, extract, chat) at DEBUG level never writes the API key to the log."""
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


TEST_PASSWORD = "correct horse battery staple"


def test_app_is_open_without_a_password(client):
    """With APP_PASSWORD empty (a local run), nothing asks for a password."""
    assert client.get("/api/documents").status_code == 200


@pytest.mark.parametrize("path", ["/api/documents", "/"])
def test_password_is_required_everywhere_when_set(client, monkeypatch, path):
    """With APP_PASSWORD set, the API and the frontend both answer 401 and make the browser ask."""
    monkeypatch.setenv("APP_PASSWORD", TEST_PASSWORD)
    response = client.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith('Basic realm="Licence Reader"')
    assert response.json() == {"error": "Password required."}


@pytest.mark.parametrize(
    ("auth", "status"),
    [
        (("anyone", TEST_PASSWORD), 200),  # any username, the right password
        (("anyone", "wrong"), 401),
        (("anyone", ""), 401),
    ],
)
def test_only_the_right_password_is_accepted(client, monkeypatch, auth, status):
    monkeypatch.setenv("APP_PASSWORD", TEST_PASSWORD)
    assert client.get("/api/documents", auth=auth).status_code == status


@pytest.mark.parametrize("header", ["Bearer abc", "Basic not-base64!", "Basic", ""])
def test_malformed_credentials_are_refused_not_crashed(client, monkeypatch, header):
    monkeypatch.setenv("APP_PASSWORD", TEST_PASSWORD)
    assert client.get("/api/documents", headers={"Authorization": header}).status_code == 401


def test_password_is_never_logged(client, monkeypatch, caplog):
    """Security: neither startup nor a refused or accepted request writes the password to the log."""
    monkeypatch.setenv("APP_PASSWORD", TEST_PASSWORD)
    with caplog.at_level(logging.DEBUG):
        main.startup_checks()
        client.get("/api/documents")
        client.get("/api/documents", auth=("anyone", TEST_PASSWORD))
    assert "Password protection: on" in caplog.text
    assert TEST_PASSWORD not in caplog.text


def test_env_example_lists_every_setting_with_spec_defaults():
    """.env.example lists every setting, with the defaults the specification gives."""
    lines = (REPO_DIR / "backend" / ".env.example").read_text(encoding="utf-8").splitlines()
    settings = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
    assert settings == {
        "OPENROUTER_API_KEY": "",
        "LLM_MODEL": "google/gemini-3.8-flash",
        "LLM_MODEL_ALT": "anthropic/claude-sonnet-5",
        "LLM_CHAT_MODEL": "",
        "MAX_UPLOAD_MB": "10",
        "APP_PASSWORD": "",
        "TESSERACT_CMD": "",
        "POPPLER_PATH": "",
    }


def test_gitignore_keeps_secrets_pii_and_build_output_out():
    """Secrets, licence documents and build output are all git-ignored."""
    entries = {line.strip() for line in (REPO_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()}
    for required in (".env", "data/", "chroma/", "samples/", "node_modules/", "__pycache__/", "dist/"):
        assert required in entries, required


def test_no_documents_or_secrets_are_tracked_by_git():
    """Security: no image, PDF, database or .env file has been committed to the repository."""
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
