from pydantic import BaseModel


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
