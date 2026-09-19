"""Document endpoints: upload, list, image, extract, and saving the reviewed form.

Uploads are validated (extension, size, magic bytes) and turned into one "working image" that
OCR, the LLM and the viewer all share, so bounding boxes line up everywhere.
"""

import asyncio
import io
import logging
import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image, ImageOps

from app.schemas import (
    CORE_FIELDS,
    DocumentSummary,
    ExtractionResult,
    ImageMeta,
    LicenceData,
    UploadResponse,
)
from app.services import extraction, ocr, rag, storage
from app.services.providers import base as providers
from app.services.providers.base import ProviderError

logger = logging.getLogger(__name__)

# Every route in this file starts with /api/documents.
router = APIRouter(prefix="/api/documents", tags=["documents"])

# extension -> kind; each kind must also match its magic bytes
_EXTENSIONS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".pdf": "pdf"}
# The first bytes every real file of that kind starts with.
_MAGIC = {"jpeg": b"\xff\xd8\xff", "png": b"\x89PNG\r\n\x1a\n", "pdf": b"%PDF-"}
_MEDIA_TYPES = {"jpeg": "image/jpeg", "png": "image/png"}
_KIND_NAMES = {"jpeg": "JPEG", "png": "PNG", "pdf": "PDF"}  # as shown in error messages
# The EXIF tag in which a camera records how the photo should be rotated.
_EXIF_ORIENTATION = 0x0112
_PDF_LONG_SIDE = 2000  # render each page so its longest side is this many pixels (bounds huge pages)
_PDF_PAGES = 2  # front and back of a two-sided licence
_PAGE_GAP = 24  # white pixels between stacked pages
_CHUNK = 1024 * 1024  # uploads are read 1 MB at a time
# Vision models cap image size (some at 5 MB); larger working images are downscaled for the LLM
# only. OCR always runs on the full-resolution working image.
_LLM_MAX_SIDE = 2048
_LLM_MAX_BYTES = 4 * 1024 * 1024


# --- helpers ----------------------------------------------------------------------------------


def poppler_path() -> str | None:
    """POPPLER_PATH if it is a real directory, else rely on poppler being on PATH.

    Falling back keeps one .env usable both locally and in the container.
    """
    path = os.getenv("POPPLER_PATH")
    if path and Path(path).is_dir():
        return path
    if path:
        logger.warning("POPPLER_PATH does not exist (%s); falling back to poppler on PATH", path)
    return None


def max_upload_mb() -> float:
    """The upload size limit in megabytes, from the MAX_UPLOAD_MB setting (default 10)."""
    return float(os.getenv("MAX_UPLOAD_MB", "10"))


def get_document_or_404(doc_id: str) -> dict:
    """The document's database row, or a 404 response when the id is unknown."""
    # Ids are UUIDs. Anything else is rejected before it reaches the database.
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
    # Keep only the last part of a path ("C:\\docs\\dl.png" -> "dl.png").
    name = re.split(r"[\\/]", filename or "")[-1]
    # Remove control characters, then cap the length.
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:120] or "document"


def _to_png(img: Image.Image) -> bytes:
    """The image encoded as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _stack(pages: list[Image.Image]) -> tuple[Image.Image, list[int]]:
    """Pages top to bottom on one white canvas; returns it and each page's top y offset."""
    # The canvas is as wide as the widest page, and as tall as all pages plus the gaps.
    width = max(p.width for p in pages)
    height = sum(p.height for p in pages) + _PAGE_GAP * (len(pages) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    offsets, y = [], 0
    for page in pages:
        canvas.paste(page.convert("RGB"), (0, y))
        offsets.append(y)
        y += page.height + _PAGE_GAP
    return canvas, offsets


def prepare_working_image(kind: str, data: bytes) -> tuple[bytes | None, str, str, int, int, list[int]]:
    """Validate the content and produce the working image.

    Returns (image_bytes or None if the original is used as-is, image_ext, media_type, width,
    height, page_offsets). A PDF's first two pages (front and back) are stacked into one image;
    page_offsets holds each page's top y, so a field's page follows from where it is found.
    """
    # A PDF is turned into one PNG: its first two pages, one above the other.
    if kind == "pdf":
        try:
            pages = convert_from_bytes(
                data,
                size=_PDF_LONG_SIDE,
                first_page=1,
                last_page=_PDF_PAGES,
                poppler_path=poppler_path(),
            )
        except PDFInfoNotInstalledError:
            # poppler is missing on the server: that is our problem (500), not the user's.
            raise HTTPException(
                500, "PDF support is unavailable: poppler is not installed or POPPLER_PATH is wrong."
            )
        except Exception:
            raise HTTPException(400, "Could not read the PDF. The file may be damaged or password-protected.")
        if not pages:
            raise HTTPException(400, "The PDF has no pages.")
        working, offsets = _stack(pages)
        return _to_png(working), ".png", "image/png", working.width, working.height, offsets

    # A JPG or PNG is opened to prove it really is an image, and is normally used as it is.
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            # Bake EXIF rotation into the pixels so OCR boxes and the displayed image agree.
            if img.getexif().get(_EXIF_ORIENTATION, 1) != 1:
                upright = ImageOps.exif_transpose(img)
                return _to_png(upright), ".png", "image/png", upright.width, upright.height, [0]
            return None, "", _MEDIA_TYPES[kind], img.width, img.height, [0]
    except Exception:
        raise HTTPException(400, "Could not read the image. The file may be damaged.")


# --- list and upload --------------------------------------------------------------------------


@router.get("", response_model=list[DocumentSummary])
def list_documents():
    """GET /api/documents - every uploaded document, newest first, for the home screen."""
    return storage.list_documents()


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile | None = File(None)):
    """Validate and store an upload. Every rejection is a 400 with a message the user can act on."""
    if file is None:
        raise HTTPException(400, "No file was uploaded. Please choose a JPG, PNG or PDF.")

    # Check 1: the file name must end in a supported extension.
    ext = os.path.splitext(file.filename or "")[1].lower()
    kind = _EXTENSIONS.get(ext)
    if kind is None:
        shown = ext or "(none)"
        raise HTTPException(400, f"Unsupported file type {shown}. Please upload a JPG, PNG or PDF.")

    # Check 2: the size.
    # Read in chunks so an oversized upload is rejected without holding all of it in memory.
    limit_mb = max_upload_mb()
    limit_bytes = int(limit_mb * 1024 * 1024)
    chunks, size = [], 0
    while chunk := await file.read(_CHUNK):
        size += len(chunk)
        if size > limit_bytes:
            raise HTTPException(400, f"File is too large. The maximum size is {limit_mb:g} MB.")
        chunks.append(chunk)
    data = b"".join(chunks)

    # Check 3: the content.
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    # The content must match the extension: a renamed file is not trusted.
    if not data.startswith(_MAGIC[kind]):
        raise HTTPException(
            400, f"The file content is not a valid {_KIND_NAMES[kind]}, although its name ends in {ext}."
        )

    # Check 4: the file must really open as an image or PDF. Decoding is slow, so it runs
    # in a worker thread and the server stays free for other requests.
    image, image_ext, media_type, width, height, page_offsets = await run_in_threadpool(
        prepare_working_image, kind, data
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
        page_offsets=page_offsets,
    )
    return {"doc_id": doc_id}


# --- the document image -----------------------------------------------------------------------


@router.get("/{doc_id}/image")
def get_image(doc_id: str):
    """GET /image - the working image, exactly as OCR saw it, for the document viewer."""
    doc = get_document_or_404(doc_id)
    path = storage.image_path(doc)
    if not path.is_file():
        raise HTTPException(404, "Document image is missing from storage")
    return FileResponse(path, media_type=doc["media_type"])


@router.get("/{doc_id}/meta", response_model=ImageMeta)
def get_meta(doc_id: str):
    """GET /meta - the working image's size in pixels, which the viewer needs to scale highlights."""
    doc = get_document_or_404(doc_id)
    return {"width": doc["width"], "height": doc["height"]}


# --- extraction -------------------------------------------------------------------------------


def image_for_llm(data: bytes, media_type: str) -> tuple[bytes, str]:
    """The image as sent to the LLM: unchanged if small enough, else downscaled to a JPEG.

    Only the LLM sees the smaller copy; OCR and the highlights keep the full working image.
    """
    with Image.open(io.BytesIO(data)) as img:
        if len(data) <= _LLM_MAX_BYTES and max(img.size) <= _LLM_MAX_SIDE:
            return data, media_type
        # JPEG has no transparency, so convert to plain RGB before shrinking and saving.
        small = img.convert("RGB")
        small.thumbnail((_LLM_MAX_SIDE, _LLM_MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


@router.post("/{doc_id}/extract", response_model=ExtractionResult)
async def extract_document(doc_id: str, background_tasks: BackgroundTasks):
    """Run OCR and the vision LLM at the same time, cross-check them, store and index the result.

    An LLM failure fails the request (502). An OCR failure does not: the fields are returned
    unconfirmed, with a warning.
    """
    doc = get_document_or_404(doc_id)
    path = storage.image_path(doc)
    if not path.is_file():
        raise HTTPException(404, "Document image is missing from storage")
    # The provider is chosen by the LLM_MODEL setting (OpenRouter, or Ollama for "ollama/...").
    try:
        provider = providers.get_provider()
    except ProviderError as e:
        raise HTTPException(502, str(e))

    image_bytes, media_type = await run_in_threadpool(image_for_llm, path.read_bytes(), doc["media_type"])
    # OCR is blocking (a thread); the LLM call is async. The wait is the slower of the two.
    # `return_exceptions=True` hands back an error as a result, so a failure of one reader
    # does not cancel the other.
    loop = asyncio.get_running_loop()
    ocr_result, llm_data = await asyncio.gather(
        loop.run_in_executor(None, ocr.run_ocr, path),
        provider.extract(image_bytes, media_type),
        return_exceptions=True,
    )

    # Without the AI's answer there is nothing to show: the request fails.
    if isinstance(llm_data, ProviderError):
        raise HTTPException(502, f"Extraction failed: {llm_data}")
    if isinstance(llm_data, BaseException):
        raise llm_data

    # Without OCR the fields can still be shown, but none of them can be confirmed.
    warnings: list[str] = []
    if isinstance(ocr_result, BaseException):
        logger.error("OCR failed for %s: %s: %s", doc_id, type(ocr_result).__name__, ocr_result)
        warnings.append("OCR could not run on this document; nothing could be cross-checked.")
        ocr_result = ocr.OcrResult(text="", words=[])

    # All eight main fields empty usually means the upload is not a licence at all.
    if all(getattr(llm_data, name).value is None for name in CORE_FIELDS):
        warnings.append("No driving licence fields were found. Is this image a driving licence?")

    # The cross-check: sets each field's confidence, highlight position and page.
    data, merge_warnings = extraction.merge(
        llm_data, ocr_result.text, words=ocr_result.words, page_offsets=storage.page_offsets(doc)
    )
    result = ExtractionResult(
        doc_id=doc_id, data=data, ocr_text=ocr_result.text, warnings=warnings + merge_warnings
    )
    storage.save_extraction(doc_id, result.model_dump_json(), ocr_words=ocr_result.words)
    # Index for chat after the response is sent; chat builds the index itself if it gets there first.
    background_tasks.add_task(rag.index_document_safely, doc_id, result)
    return result


def _load_extraction(doc_id: str) -> ExtractionResult:
    """The stored extraction, or a 404 when the document is unknown or not extracted yet."""
    get_document_or_404(doc_id)
    raw = storage.get_extraction(doc_id)
    if raw is None:
        raise HTTPException(404, "This document has not been extracted yet.")
    return ExtractionResult.model_validate_json(raw)


@router.get("/{doc_id}/extract", response_model=ExtractionResult)
def get_extraction(doc_id: str):
    """GET /extract - the saved result, so reopening a document does not call the AI again."""
    return _load_extraction(doc_id)


# --- saving the reviewed form -----------------------------------------------------------------


@router.put("/{doc_id}/data", response_model=LicenceData)
def save_data(doc_id: str, data: LicenceData, background_tasks: BackgroundTasks):
    """PUT /data - save the user's edits. Invalid input (such as a bad date) is a 422."""
    result = _load_extraction(doc_id)
    try:
        result.data = extraction.clean_user_data(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    # `ocr_words` is not passed: the OCR results stay as they were.
    storage.save_extraction(doc_id, result.model_dump_json())
    # Keep chat grounded in the corrected field values.
    background_tasks.add_task(rag.index_document_safely, doc_id, result)
    return result.data
