"""Step 1 - backend skeleton: upload validation, working image, meta, document list, 404s."""

import io
import uuid

import pytest
from conftest import PNG_MAGIC, image_bytes, make_licence, pdf_bytes, poppler_available, upload
from PIL import Image

pytestmark = [pytest.mark.phase1, pytest.mark.step1]


def assert_error(response, status):
    assert response.status_code == status, response.text
    body = response.json()
    assert isinstance(body.get("error"), str) and body["error"].strip()
    return body["error"]


def test_upload_validation(client, monkeypatch):
    """Spec test 5: .txt -> 400 with message; oversized -> 400; valid png -> 201 with doc_id."""
    message = assert_error(upload(client, b"hello", "notes.txt", "text/plain"), 400)
    assert "JPG, PNG or PDF" in message

    monkeypatch.setenv("MAX_UPLOAD_MB", "0.01")  # ~10 KB
    message = assert_error(upload(client, PNG_MAGIC + b"\0" * 20_000, "big.png"), 400)
    assert "too large" in message.lower()
    monkeypatch.setenv("MAX_UPLOAD_MB", "10")

    response = upload(client, image_bytes(), "licence.png")
    assert response.status_code == 201
    uuid.UUID(response.json()["doc_id"])


@pytest.mark.parametrize(
    "filename, content",
    [
        ("fake.png", b"this is text, not a png"),  # extension ok, magic bytes wrong
        ("photo.jpg", image_bytes(fmt="PNG")),  # a PNG named .jpg
        ("scan.pdf", image_bytes(fmt="PNG")),  # a PNG named .pdf
        ("empty.png", b""),
        ("broken.png", PNG_MAGIC + b"garbage"),  # valid magic, undecodable
        ("noextension", image_bytes()),
        ("anim.gif", b"GIF89a" + b"\0" * 20),
    ],
)
def test_rejects_invalid_uploads(client, filename, content):
    assert_error(upload(client, content, filename, "application/octet-stream"), 400)


def test_rejects_request_without_file(client):
    assert_error(client.post("/api/documents"), 400)


@pytest.mark.parametrize("fmt, filename, media_type", [("PNG", "a.png", "image/png"), ("JPEG", "b.jpeg", "image/jpeg")])
def test_image_upload_is_served_back_unchanged(client, fmt, filename, media_type):
    content = image_bytes(size=(321, 123), fmt=fmt)
    doc_id = upload(client, content, filename, media_type).json()["doc_id"]

    image = client.get(f"/api/documents/{doc_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == media_type
    assert image.content == content
    assert client.get(f"/api/documents/{doc_id}/meta").json() == {"width": 321, "height": 123}


def test_exif_rotation_is_baked_into_working_image(client):
    # Orientation 6 = rotate 90 degrees: a 300x100 JPEG displays as 100x300.
    content = image_bytes(size=(300, 100), fmt="JPEG", exif_orientation=6)
    doc_id = upload(client, content, "phone.jpg", "image/jpeg").json()["doc_id"]

    assert client.get(f"/api/documents/{doc_id}/meta").json() == {"width": 100, "height": 300}
    served = Image.open(io.BytesIO(client.get(f"/api/documents/{doc_id}/image").content))
    assert served.size == (100, 300)


@pytest.mark.skipif(not poppler_available(), reason="poppler (pdftoppm) not installed")
def test_pdf_first_page_becomes_working_png(client):
    doc_id = upload(client, pdf_bytes(size=(850, 1100)), "licence.pdf", "application/pdf").json()["doc_id"]

    meta = client.get(f"/api/documents/{doc_id}/meta").json()
    assert max(meta["width"], meta["height"]) == 2000  # long side capped for huge pages
    assert meta["height"] > meta["width"]  # portrait page stays portrait
    image = client.get(f"/api/documents/{doc_id}/image")
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(PNG_MAGIC)


def test_document_list(client):
    assert client.get("/api/documents").json() == []
    first = upload(client, image_bytes(), "first.png").json()["doc_id"]
    second = upload(client, image_bytes(), "second.png").json()["doc_id"]

    docs = client.get("/api/documents").json()
    assert [d["doc_id"] for d in docs] == [second, first]  # newest first
    for d in docs:
        assert set(d) == {"doc_id", "filename_label", "uploaded_at", "has_extraction"}
        assert d["has_extraction"] is False
    assert docs[1]["filename_label"] == "first.png"


def test_stored_under_server_uuid_never_client_filename(client):
    from app.services import storage

    doc_id = upload(client, image_bytes(), "../../evil name.png").json()["doc_id"]

    assert [p.name for p in storage.uploads_dir().iterdir()] == [f"{doc_id}.png"]
    label = client.get("/api/documents").json()[0]["filename_label"]
    assert label == "evil name.png"  # display label only, path parts stripped


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", "/image"),
        ("get", "/meta"),
        ("get", "/extract"),
        ("post", "/extract"),
        ("put", "/data"),
        ("post", "/chat"),
    ],
)
@pytest.mark.parametrize("doc_id", [str(uuid.uuid4()), "not-a-uuid"])
def test_unknown_document_is_404_json(client, method, path, doc_id):
    bodies = {"/chat": {"question": "hi"}, "/data": make_licence().model_dump()}
    kwargs = {"json": bodies[path]} if path in bodies else {}
    response = getattr(client, method)(f"/api/documents/{doc_id}{path}", **kwargs)
    assert_error(response, 404)


def test_unknown_api_route_is_json_error(client):
    assert_error(client.get("/api/nope"), 404)
    assert_error(client.delete("/api/documents"), 405)
