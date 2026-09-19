"""Request and response models shared by the API, the providers and the services."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int  # image pixel coordinates of the working image


class FieldValue(BaseModel):
    """One extracted value with its evidence: the printed text, a confidence and where it is."""

    value: str | None
    source_text: str | None  # verbatim text as printed on the document
    confidence: Literal["high", "review"]
    bbox: Box | None  # null when no OCR match found
    page: int = 1


class LicenceData(BaseModel):
    """The licence as the LLM returns it and as the user edits it."""

    full_name: FieldValue
    licence_number: FieldValue
    date_of_birth: FieldValue  # value normalized to YYYY-MM-DD; source_text keeps printed form
    date_of_issue: FieldValue
    date_of_expiry: FieldValue
    address: FieldValue
    vehicle_classes: FieldValue  # comma-joined if multiple
    issuing_authority: FieldValue
    other_fields: dict[str, FieldValue]  # blood group, relation name, reference numbers, state, etc.


class ExtractionResult(BaseModel):
    """What /extract returns and what is stored: fields, raw OCR text and review warnings."""

    doc_id: str
    data: LicenceData
    ocr_text: str
    warnings: list[str]


# Fixed LicenceData fields, in display order (other_fields is dynamic).
CORE_FIELDS: tuple[str, ...] = (
    "full_name",
    "licence_number",
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "address",
    "vehicle_classes",
    "issuing_authority",
)
DATE_FIELDS: frozenset[str] = frozenset({"date_of_birth", "date_of_issue", "date_of_expiry"})


class ChatRequest(BaseModel):
    question: str
    today: date | None = None  # the browser's local date, for "days until expiry" answers


class ChatSource(BaseModel):
    text: str
    # "calculated" = date arithmetic worked out by the app, not text from the document
    origin: Literal["ocr_text", "extracted_fields", "calculated"]
    bbox: Box | None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


class DocumentSummary(BaseModel):
    doc_id: str
    filename_label: str
    uploaded_at: str
    has_extraction: bool


class UploadResponse(BaseModel):
    doc_id: str


class ImageMeta(BaseModel):
    width: int
    height: int
