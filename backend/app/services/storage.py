"""SQLite metadata + on-disk file persistence.

Layout under the data directory (default ./data, relative to the working dir):
  app.db             SQLite database
  uploads/<uuid>.*   uploaded originals and working images (server-generated names only)
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_data_dir = Path("data")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename_label  TEXT NOT NULL,     -- display label only; never used on disk
    original_name   TEXT NOT NULL,     -- stored file name of the upload
    image_name      TEXT NOT NULL,     -- stored file name of the working image
    media_type      TEXT NOT NULL,     -- media type of the working image
    page_number     INTEGER NOT NULL,  -- source page of the working image (PDFs: 1)
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    uploaded_at     TEXT NOT NULL,
    extraction      TEXT               -- JSON ExtractionResult; NULL until extracted
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


def uploads_dir() -> Path:
    return _data_dir / "uploads"


@contextmanager
def _connect():
    conn = sqlite3.connect(_data_dir / "app.db")
    conn.row_factory = sqlite3.Row
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
    page_number: int = 1,
) -> str:
    """Persist an upload. `image=None` means the original itself is the working image."""
    doc_id = str(uuid.uuid4())
    original_name = f"{doc_id}{original_ext}"
    (uploads_dir() / original_name).write_bytes(original)
    if image is None:
        image_name = original_name
    else:
        image_name = f"{doc_id}.page{page_number}{image_ext}"
        (uploads_dir() / image_name).write_bytes(image)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename_label, original_name, image_name, media_type,"
            " page_number, width, height, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                filename_label,
                original_name,
                image_name,
                media_type,
                page_number,
                width,
                height,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return doc_id


def get_document(doc_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, filename_label, uploaded_at, extraction IS NOT NULL AS has_extraction"
            " FROM documents ORDER BY uploaded_at DESC, rowid DESC"
        ).fetchall()
    return [{**dict(r), "has_extraction": bool(r["has_extraction"])} for r in rows]


def image_path(doc: dict) -> Path:
    return uploads_dir() / doc["image_name"]
