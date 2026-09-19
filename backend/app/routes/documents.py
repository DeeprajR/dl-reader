import io
import os
import re
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image, ImageOps

from app.schemas import DocumentSummary, ImageMeta, UploadResponse
from app.services import storage

router = APIRouter(prefix="/api/documents", tags=["documents"])

# extension -> kind; each kind must also match its magic bytes
_EXTENSIONS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".pdf": "pdf"}
_MAGIC = {"jpeg": b"\xff\xd8\xff", "png": b"\x89PNG\r\n\x1a\n", "pdf": b"%PDF-"}
_MEDIA_TYPES = {"jpeg": "image/jpeg", "png": "image/png"}
_KIND_NAMES = {"jpeg": "JPEG", "png": "PNG", "pdf": "PDF"}
_EXIF_ORIENTATION = 0x0112
_PDF_DPI = 200
_CHUNK = 1024 * 1024


def max_upload_mb() -> float:
    return float(os.getenv("MAX_UPLOAD_MB", "10"))


def get_document_or_404(doc_id: str) -> dict:
    try:
        uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(404, "Document not found")
    doc = storage.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


def _filename_label(filename: str | None) -> str:
    """Display label derived from the client filename. Never used as a path."""
    name = re.split(r"[\\/]", filename or "")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:120] or "document"


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _prepare_working_image(kind: str, data: bytes) -> tuple[bytes | None, str, str, int, int]:
    """Validate the content and produce the working image.

    Returns (image_bytes or None if the original is used as-is, image_ext, media_type, width, height).
    """
    if kind == "pdf":
        try:
            pages = convert_from_bytes(
                data,
                dpi=_PDF_DPI,
                first_page=1,
                last_page=1,
                poppler_path=os.getenv("POPPLER_PATH") or None,
            )
        except PDFInfoNotInstalledError:
            raise HTTPException(
                500, "PDF support is unavailable: poppler is not installed or POPPLER_PATH is wrong."
            )
        except Exception:
            raise HTTPException(400, "Could not read the PDF. The file may be damaged or password-protected.")
        if not pages:
            raise HTTPException(400, "The PDF has no pages.")
        page = pages[0]
        return _to_png(page), ".png", "image/png", page.width, page.height

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            # Bake EXIF rotation into the pixels so OCR boxes and the displayed image agree.
            if img.getexif().get(_EXIF_ORIENTATION, 1) != 1:
                upright = ImageOps.exif_transpose(img)
                return _to_png(upright), ".png", "image/png", upright.width, upright.height
            return None, "", _MEDIA_TYPES[kind], img.width, img.height
    except Exception:
        raise HTTPException(400, "Could not read the image. The file may be damaged.")


@router.get("", response_model=list[DocumentSummary])
def list_documents():
    return storage.list_documents()


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile | None = File(None)):
    if file is None:
        raise HTTPException(400, "No file was uploaded. Please choose a JPG, PNG or PDF.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    kind = _EXTENSIONS.get(ext)
    if kind is None:
        shown = ext or "(none)"
        raise HTTPException(400, f"Unsupported file type {shown}. Please upload a JPG, PNG or PDF.")

    limit_mb = max_upload_mb()
    limit_bytes = int(limit_mb * 1024 * 1024)
    chunks, size = [], 0
    while chunk := await file.read(_CHUNK):
        size += len(chunk)
        if size > limit_bytes:
            raise HTTPException(400, f"File is too large. The maximum size is {limit_mb:g} MB.")
        chunks.append(chunk)
    data = b"".join(chunks)

    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if not data.startswith(_MAGIC[kind]):
        raise HTTPException(
            400, f"The file content is not a valid {_KIND_NAMES[kind]}, although its name ends in {ext}."
        )

    image, image_ext, media_type, width, height = await run_in_threadpool(
        _prepare_working_image, kind, data
    )
    doc_id = storage.save_document(
        filename_label=_filename_label(file.filename),
        original=data,
        original_ext=".jpg" if kind == "jpeg" else ext,
        image=image,
        image_ext=image_ext,
        media_type=media_type,
        width=width,
        height=height,
        page_number=1,
    )
    return {"doc_id": doc_id}


@router.get("/{doc_id}/image")
def get_image(doc_id: str):
    doc = get_document_or_404(doc_id)
    path = storage.image_path(doc)
    if not path.is_file():
        raise HTTPException(404, "Document image is missing from storage")
    return FileResponse(path, media_type=doc["media_type"])


@router.get("/{doc_id}/meta", response_model=ImageMeta)
def get_meta(doc_id: str):
    doc = get_document_or_404(doc_id)
    return {"width": doc["width"], "height": doc["height"]}
