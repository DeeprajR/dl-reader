"""Step 1 - backend skeleton: upload validation, working image, meta, document list, 404s."""

import io
import uuid

import pytest
from conftest import PNG_MAGIC, image_bytes, make_licence, pdf_bytes, poppler_available, upload
from PIL import Image

pytestmark = [pytest.mark.phase1, pytest.mark.step1]


def assert_error(response, status):
    """The response must have this status and the API's error shape, {"error": "message"}. Returns the message."""
    assert response.status_code == status, response.text
    body = response.json()
    assert isinstance(body.get("error"), str) and body["error"].strip()
    return body["error"]


def test_upload_validation(client, monkeypatch):
    """Spec test 5: .txt -> 400 with message; oversized -> 400; valid png -> 201 with doc_id."""
    message = assert_error(upload(client, b"hello", "notes.txt", "text/plain"), 400)
    assert "JPG, PNG or PDF" in message

    # Lower the limit instead of building a real 10 MB file.
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
    """Every way a file can be wrong gives a 400: wrong content, wrong extension, empty, damaged, unsupported type."""
    assert_error(upload(client, content, filename, "application/octet-stream"), 400)


def test_rejects_request_without_file(client):
    """A request with no file at all gets a clear 400, not a server error."""
    assert_error(client.post("/api/documents"), 400)


@pytest.mark.parametrize("fmt, filename, media_type", [("PNG", "a.png", "image/png"), ("JPEG", "b.jpeg", "image/jpeg")])
def test_image_upload_is_served_back_unchanged(client, fmt, filename, media_type):
    """A normal JPG or PNG is stored byte for byte, and /meta reports its real size."""
    content = image_bytes(size=(321, 123), fmt=fmt)
    doc_id = upload(client, content, filename, media_type).json()["doc_id"]

    image = client.get(f"/api/documents/{doc_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == media_type
    assert image.content == content
    assert client.get(f"/api/documents/{doc_id}/meta").json() == {"width": 321, "height": 123}


def test_exif_rotation_is_baked_into_working_image(client):
    """A phone photo tagged "rotate me" is turned upright, so OCR boxes and the displayed image agree."""
    # Orientation 6 = rotate 90 degrees: a 300x100 JPEG displays as 100x300.
    content = image_bytes(size=(300, 100), fmt="JPEG", exif_orientation=6)
    doc_id = upload(client, content, "phone.jpg", "image/jpeg").json()["doc_id"]

    assert client.get(f"/api/documents/{doc_id}/meta").json() == {"width": 100, "height": 300}
    served = Image.open(io.BytesIO(client.get(f"/api/documents/{doc_id}/image").content))
    assert served.size == (100, 300)


@pytest.mark.skipif(not poppler_available(), reason="poppler (pdftoppm) not installed")
def test_pdf_first_page_becomes_working_png(client):
    """A PDF becomes a PNG whose long side is 2000 px."""
    doc_id = upload(client, pdf_bytes(size=(850, 1100)), "licence.pdf", "application/pdf").json()["doc_id"]

    meta = client.get(f"/api/documents/{doc_id}/meta").json()
    assert max(meta["width"], meta["height"]) == 2000  # long side capped for huge pages
    assert meta["height"] > meta["width"]  # portrait page stays portrait
    image = client.get(f"/api/documents/{doc_id}/image")
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(PNG_MAGIC)


def test_document_list(client):
    """The list is empty at first, then newest first, with exactly the fields the home screen needs."""
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
    """Security: a hostile file name ("../../evil") never reaches the disk. Files are named after the document id."""
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
    """Every endpoint answers 404 JSON for an unknown id and for an id that is not a UUID at all."""
    # The two endpoints that need a body get a valid one, so the 404 is about the id and nothing else.
    bodies = {"/chat": {"question": "hi"}, "/data": make_licence().model_dump()}
    kwargs = {"json": bodies[path]} if path in bodies else {}
    response = getattr(client, method)(f"/api/documents/{doc_id}{path}", **kwargs)
    assert_error(response, 404)


def test_unknown_api_route_is_json_error(client):
    """Unknown routes and wrong methods also answer in the API's JSON error shape, not HTML."""
    assert_error(client.get("/api/nope"), 404)
    assert_error(client.delete("/api/documents"), 405)


def multipage_pdf(n, size=(850, 1100)):
    """A blank PDF with `n` pages, built in memory."""
    buf = io.BytesIO()
    pages = [Image.new("RGB", size, "white") for _ in range(n)]
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


@pytest.mark.skipif(not poppler_available(), reason="poppler (pdftoppm) not installed")
def test_two_page_pdf_stacks_front_and_back(client):
    """Pages 1 and 2 are stacked into one image with a 24 px gap, and any further pages are ignored."""
    one = client.get(f"/api/documents/{upload(client, multipage_pdf(1), 'a.pdf', 'application/pdf').json()['doc_id']}/meta").json()
    doc_id = upload(client, multipage_pdf(2), "b.pdf", "application/pdf").json()["doc_id"]
    two = client.get(f"/api/documents/{doc_id}/meta").json()
    assert two["width"] == one["width"]
    assert two["height"] == 2 * one["height"] + 24  # page 2 below page 1, 24px gap

    doc_id = upload(client, multipage_pdf(3), "c.pdf", "application/pdf").json()["doc_id"]
    assert client.get(f"/api/documents/{doc_id}/meta").json() == two  # only front and back are read
