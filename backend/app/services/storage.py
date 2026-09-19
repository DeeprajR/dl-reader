"""Storage: a small SQLite database for the details, and plain files for the documents.

Layout under the data directory (default ./data, relative to the working dir):
  app.db             SQLite database
  uploads/<uuid>.*   uploaded originals and working images (server-generated names only)

Files are always named after the document's random id, never after the name the user's file
had. That way a crafted file name can never reach the file system.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Where everything is kept. `init` can point it somewhere else; the tests use a temporary folder.
_data_dir = Path("data")

# One table, one row per uploaded document.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename_label  TEXT NOT NULL,     -- display label only; never used on disk
    original_name   TEXT NOT NULL,     -- stored file name of the upload
    image_name      TEXT NOT NULL,     -- stored file name of the working image
    media_type      TEXT NOT NULL,     -- media type of the working image
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    uploaded_at     TEXT NOT NULL,
    extraction      TEXT,              -- JSON ExtractionResult; NULL until extracted
    ocr_words       TEXT               -- JSON Tesseract word boxes from the last extraction
)
"""


def init(data_dir: Path | str | None = None) -> None:
    """Create the data directories and database schema. Safe to call repeatedly."""
    global _data_dir
    if data_dir is not None:
        _data_dir = Path(data_dir)
    uploads_dir().mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(_SCHEMA)
        # "CREATE TABLE IF NOT EXISTS" leaves an existing table as it is, so a database made
        # by an older version is brought up to date here.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
        if "ocr_words" not in columns:  # databases created before word boxes were stored
            conn.execute("ALTER TABLE documents ADD COLUMN ocr_words TEXT")
        for unused in ("page_number", "page_offsets"):  # columns that were stored but never read
            if unused in columns:
                conn.execute(f"ALTER TABLE documents DROP COLUMN {unused}")


def uploads_dir() -> Path:
    """The folder that holds the uploaded files and the working images."""
    return _data_dir / "uploads"


@contextmanager
def _connect():
    """A database connection for one `with` block: commits at the end, and always closes.

    If the block raises, nothing is committed.
    """
    conn = sqlite3.connect(_data_dir / "app.db")
    conn.row_factory = sqlite3.Row  # rows can be read by column name
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_document(
    *,
    filename_label: str,
    original: bytes,
    original_ext: str,
    image: bytes | None,
    image_ext: str,
    media_type: str,
    width: int,
    height: int,
) -> str:
    """Persist an upload. `image=None` means the original itself is the working image."""
    doc_id = str(uuid.uuid4())
    # The uploaded file is kept as it arrived.
    original_name = f"{doc_id}{original_ext}"
    (uploads_dir() / original_name).write_bytes(original)
    # A separate working image exists only when one had to be made: a PDF turned into an
    # image, or a photo that was rotated upright.
    if image is None:
        image_name = original_name
    else:
        image_name = f"{doc_id}.working{image_ext}"
        (uploads_dir() / image_name).write_bytes(image)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename_label, original_name, image_name, media_type,"
            " width, height, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                filename_label,
                original_name,
                image_name,
                media_type,
                width,
                height,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return doc_id


def get_document(doc_id: str) -> dict | None:
    """The document's whole row as a dict, or None when there is no such document."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    """All documents for the home screen, newest first (without their extraction data)."""
    with _connect() as conn:
        # `rowid` breaks ties between documents uploaded within the same second.
        rows = conn.execute(
            "SELECT doc_id, filename_label, uploaded_at, extraction IS NOT NULL AS has_extraction"
            " FROM documents ORDER BY uploaded_at DESC, rowid DESC"
        ).fetchall()
    # SQLite has no boolean type: it returns 0 or 1, which is turned into False or True here.
    return [{**dict(r), "has_extraction": bool(r["has_extraction"])} for r in rows]


def image_path(doc: dict) -> Path:
    """Where the document's working image is on disk."""
    return uploads_dir() / doc["image_name"]


def save_extraction(doc_id: str, extraction_json: str, ocr_words: list[dict] | None = None) -> None:
    """Persist the extraction; `ocr_words` is replaced only when given (user edits keep it)."""
    with _connect() as conn:
        if ocr_words is None:
            conn.execute("UPDATE documents SET extraction = ? WHERE doc_id = ?", (extraction_json, doc_id))
        else:
            conn.execute(
                "UPDATE documents SET extraction = ?, ocr_words = ? WHERE doc_id = ?",
                (extraction_json, json.dumps(ocr_words), doc_id),
            )


def get_ocr_words(doc_id: str) -> list[dict]:
    """The word positions from the last extraction, or [] when there are none."""
    with _connect() as conn:
        row = conn.execute("SELECT ocr_words FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return json.loads(row["ocr_words"]) if row and row["ocr_words"] else []


def get_extraction(doc_id: str) -> str | None:
    """The stored ExtractionResult as JSON text, or None when the document is not extracted yet."""
    with _connect() as conn:
        row = conn.execute("SELECT extraction FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return row["extraction"] if row else None
