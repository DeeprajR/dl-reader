"""Shared fixtures and the phase/step pass-fail gates.

No real LLM or network calls: extraction uses a FakeProvider injected via the provider factory,
any real OpenRouter completion raises, embeddings come from a deterministic hashing embedder and
ChromaDB runs in memory.

Phase / step gates
  Every test module is marked with its phase and build step, so each can be checked alone:
      pytest                 everything
      pytest -m phase1       Phase 1 gate (steps 1-5): must pass
      pytest -m step4        one build step
  Phase 2 acceptance tests are written ahead of the code. A Phase 2 step not yet listed in
  PHASE2_STEPS_DONE has its tests collected as strict xfail ("pending"): they define what
  passing means without failing the suite. When a step is built, add it to PHASE2_STEPS_DONE
  and its tests become binding pass/fail. (A pending test that already passes is reported as
  a failure, so a finished step cannot go unmarked.)
"""

import hashlib
import io
import math
import os
import re
import shutil
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent

# Set before app.main is imported: load_dotenv never overrides existing variables, so the
# developer's real .env (and API key) is never used by the tests.
TEST_API_KEY = "sk-or-test-key-not-real-0123456789"
os.environ["OPENROUTER_API_KEY"] = TEST_API_KEY
os.environ["LLM_MODEL"] = "test/fake-vision-model"
os.environ["LLM_MODEL_ALT"] = "test/fake-alt-model"
os.environ["LLM_CHAT_MODEL"] = ""
os.environ["MAX_UPLOAD_MB"] = "10"

from PIL import Image  # noqa: E402

import app.main  # noqa: E402,F401  loads .env (TESSERACT_CMD, POPPLER_PATH) before skip checks run
from app.schemas import FieldValue, LicenceData  # noqa: E402

# Phase 2 steps that are built; their tests must pass. See module docstring.
PHASE2_STEPS_DONE: set[int] = {6, 8, 9}
FIRST_PHASE2_STEP = 6


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in item.iter_markers():
            step = re.fullmatch(r"step(\d+)", marker.name)
            if step and int(step[1]) >= FIRST_PHASE2_STEP and int(step[1]) not in PHASE2_STEPS_DONE:
                item.add_marker(
                    pytest.mark.xfail(strict=True, reason=f"pending: Phase 2 step {step[1]} not built yet")
                )


# --- canned document -------------------------------------------------------------------------

IMAGE_SIZE = (800, 500)
CANNED_LINES = [
    "DRIVING LICENCE",
    "DL No: MH12 20190001234",
    "Name : JOHN DOE",
    "DOB : 12-08-1990 DOI : 16-06-2019",
    "Valid Till : 15-06-2034",
    "Address : 12 High Street,",
    "Pune",
    "Class : LMV, MCWG",
    "Issuing Authority : RTO, Pune",
    "BG : O+",
]


def words_from_lines(lines, x0=20, y0=20, line_h=40, char_w=12, h=24):
    """Synthetic Tesseract word list: one row per line, boxes laid out left to right."""
    words = []
    for i, line in enumerate(lines):
        x = x0
        for token in line.split():
            width = len(token) * char_w
            words.append(
                {"text": token, "left": x, "top": y0 + i * line_h, "width": width, "height": h, "conf": 95.0, "line": i}
            )
            x += width + char_w
    return words


CANNED_WORDS = words_from_lines(CANNED_LINES)
CANNED_OCR_TEXT = "\n".join(" ".join(line.split()) for line in CANNED_LINES)


def fv(value, source=...):
    """FieldValue as a provider returns it (confidence/bbox are set later by the merge)."""
    return FieldValue(value=value, source_text=value if source is ... else source, confidence="review", bbox=None)


def make_licence(**overrides) -> LicenceData:
    fields = dict(
        full_name=fv("JOHN DOE"),
        licence_number=fv("MH12 20190001234"),
        date_of_birth=fv("1990-08-12", "12-08-1990"),
        date_of_issue=fv("2019-06-16", "16-06-2019"),
        date_of_expiry=fv("2034-06-15", "15-06-2034"),
        address=fv("12 High Street, Pune", "12 High Street,\nPune"),
        vehicle_classes=fv("LMV, MCWG"),
        issuing_authority=fv("RTO, Pune"),
        other_fields={"blood_group": fv("O+")},
    )
    fields.update(overrides)
    return LicenceData(**fields)


class FakeProvider:
    """ExtractionProvider returning canned LicenceData (or raising `error`)."""

    def __init__(self, data: LicenceData | None = None, error: Exception | None = None):
        self.data = data or make_licence()
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    async def extract(self, image_bytes: bytes, media_type: str) -> LicenceData:
        self.calls.append((image_bytes, media_type))
        if self.error:
            raise self.error
        return self.data.model_copy(deep=True)


# --- file builders ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def image_bytes(size=IMAGE_SIZE, fmt="PNG", color="white", exif_orientation=None) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        img.save(buf, fmt, exif=exif)
    else:
        img.save(buf, fmt)
    return buf.getvalue()


def pdf_bytes(size=(850, 1100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PDF")
    return buf.getvalue()


def upload(client, content: bytes, filename="licence.png", mime="image/png"):
    return client.post("/api/documents", files={"file": (filename, content, mime)})


def poppler_available() -> bool:
    poppler_dir = os.getenv("POPPLER_PATH")
    if poppler_dir and Path(poppler_dir).is_dir():
        return shutil.which("pdftoppm", path=poppler_dir) is not None
    return shutil.which("pdftoppm") is not None


# --- fake embeddings -------------------------------------------------------------------------

EMBED_DIM = 256


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing embedder: texts sharing words are close."""
    vectors = []
    for text in texts:
        v = [0.0] * EMBED_DIM
        v[0] = 0.1  # never a zero vector
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[1 + int(hashlib.md5(token.encode()).hexdigest(), 16) % (EMBED_DIM - 1)] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        vectors.append([x / norm for x in v])
    return vectors


# --- fixtures --------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chroma_client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


@pytest.fixture(autouse=True)
def offline(monkeypatch, chroma_client):
    """Every test: real LLM calls fail loudly, embeddings are fake, Chroma is in memory."""
    from app.services import rag
    from app.services.providers import openrouter

    async def no_real_llm(*args, **kwargs):
        raise AssertionError("tests must not call the real LLM")

    monkeypatch.setattr(openrouter, "complete", no_real_llm)
    monkeypatch.setattr(rag, "complete", no_real_llm)
    monkeypatch.setattr(rag, "_embed", fake_embed)
    monkeypatch.setattr(rag, "warm_up", lambda: None)
    monkeypatch.setattr(rag, "_client", chroma_client)


@pytest.fixture
def client(tmp_path):
    """TestClient on an isolated data directory (startup checks + lifespan run)."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import storage

    storage.init(tmp_path / "data")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_provider(monkeypatch):
    from app.services.providers import base

    provider = FakeProvider()
    monkeypatch.setattr(base, "get_provider", lambda model=None: provider)
    return provider


@pytest.fixture
def fake_ocr(monkeypatch):
    from app.services import ocr

    result = ocr.OcrResult(text=CANNED_OCR_TEXT, words=[dict(w) for w in CANNED_WORDS])
    monkeypatch.setattr(ocr, "run_ocr", lambda path: result)
    return result


@pytest.fixture
def uploaded(client):
    """doc_id of an uploaded (not yet extracted) IMAGE_SIZE png."""
    response = upload(client, image_bytes())
    assert response.status_code == 201, response.text
    return response.json()["doc_id"]


@pytest.fixture
def extracted(client, uploaded, fake_provider, fake_ocr):
    """doc_id of a document extracted with the FakeProvider and canned OCR."""
    response = client.post(f"/api/documents/{uploaded}/extract")
    assert response.status_code == 200, response.text
    return uploaded
