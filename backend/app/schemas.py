"""Data models for the API.

Every request and response of the backend is one of these models, so this file is the contract
between the frontend, the routes, the AI providers and the services.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Box(BaseModel):
    """A rectangle on the document image, used to draw a highlight.

    All four numbers are pixels of the working image (the image the viewer shows).
    `x` and `y` are the top-left corner.
    """

    x: int
    y: int
    w: int
    h: int


class FieldValue(BaseModel):
    """One value read from the licence, together with the evidence for it."""

    # The cleaned-up value shown in the form. None means the field is not on the licence.
    value: str | None
    # The text exactly as printed on the card, copied by the AI model. It is what gets
    # checked against OCR, and what the form shows as the field's source.
    source_text: str | None
    # "high" means OCR confirmed the source text. "review" means it could not be confirmed,
    # and the form marks the field "Please verify".
    confidence: Literal["high", "review"]
    # Where the source text is on the image. None when OCR could not find it.
    bbox: Box | None
    # The PDF page the value is on: 1 for the front, 2 for the back. Images are always page 1.
    page: int = 1


class LicenceData(BaseModel):
    """All the fields of one licence.

    The AI provider returns this model, the user edits it in the form, and PUT /data saves it.
    """

    full_name: FieldValue
    licence_number: FieldValue
    # Dates are stored as YYYY-MM-DD in `value`; `source_text` keeps the printed form
    # (for example "15-06-2034").
    date_of_birth: FieldValue
    date_of_issue: FieldValue
    date_of_expiry: FieldValue
    address: FieldValue
    # Every vehicle class on the licence, joined with commas (for example "LMV, MCWG").
    vehicle_classes: FieldValue
    issuing_authority: FieldValue
    # Everything else printed on the card, keyed by a snake_case name: blood group, relative's
    # name, reference numbers, and each vehicle class's own dates (for example "lmv_valid_till").
    other_fields: dict[str, FieldValue]


class ExtractionResult(BaseModel):
    """The response of POST /extract. It is also stored, so reopening a document is instant."""

    doc_id: str
    data: LicenceData
    # The full text OCR read from the image. The chat searches it.
    ocr_text: str
    # Messages for the reviewer, for example "Date of issue is not before the expiry date."
    warnings: list[str]


# The fixed fields of LicenceData, in the order the form shows them. `other_fields` is not
# listed because its keys differ from licence to licence.
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

# The core fields that hold a date, and are therefore normalised to YYYY-MM-DD.
DATE_FIELDS: frozenset[str] = frozenset({"date_of_birth", "date_of_issue", "date_of_expiry"})


class ChatRequest(BaseModel):
    """The body of POST /chat."""

    question: str
    # Today's date in the user's time zone, sent by the browser. It is needed for answers like
    # "days until expiry", because the server's clock may be in another time zone.
    today: date | None = None


class ChatSource(BaseModel):
    """One passage that a chat answer is based on."""

    text: str
    # Where the passage came from:
    #   "ocr_text"         - text OCR read from the image
    #   "extracted_fields" - a field of the form
    #   "calculated"       - date arithmetic done by the app (for example days until expiry)
    origin: Literal["ocr_text", "extracted_fields", "calculated"]
    # Where the passage is on the image, so it can be highlighted. None when it was not found.
    bbox: Box | None


class ChatResponse(BaseModel):
    """The response of POST /chat."""

    # The answer, or exactly "The document does not contain this information."
    answer: str
    # The passages the answer is based on. Empty when the answer is the refusal.
    sources: list[ChatSource]


class DocumentSummary(BaseModel):
    """One row of the document list on the home screen (GET /documents)."""

    doc_id: str
    # The uploaded file's name, cleaned for display. Files are stored under `doc_id`, never
    # under this name.
    filename_label: str
    # When the document was uploaded, as an ISO 8601 timestamp in UTC.
    uploaded_at: str
    # True once the document has been read, so the list can show "Extracted".
    has_extraction: bool


class UploadResponse(BaseModel):
    """The response of POST /documents: the id of the new document."""

    doc_id: str


class ImageMeta(BaseModel):
    """The size of the working image in pixels (GET /meta). The viewer scales highlights with it."""

    width: int
    height: int
