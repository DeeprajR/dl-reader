"""Extraction provider Protocol, factory, and the prompt/parsing shared by all providers."""

import json
import os
import re
from typing import Protocol

from app.schemas import CORE_FIELDS, FieldValue, LicenceData

DEFAULT_MODEL = "google/gemini-3.8-flash"

EXTRACTION_SYSTEM_PROMPT = """
You are a precise document data extraction system. You will receive an image of a driving licence.
Extract ONLY information that is actually visible in the image.
Rules:
1. For every field, also return source_text: the exact verbatim text as printed on the document that you used for that value.
2. If a field is not visible or not present, set value and source_text to null. NEVER guess, infer, or fabricate. A wrong licence number is worse than a null.
3. Normalize dates in "value" to YYYY-MM-DD but keep source_text exactly as printed.
4. licence_number: preserve exact characters, spacing and case as printed.
5. Put any additional identifiable fields (blood group, relation name, reference numbers, state, country) into other_fields with descriptive snake_case keys.
6. Respond with ONLY a JSON object matching the provided schema. No prose, no markdown fences.
""".strip()

_FIELD = '{"value": string | null, "source_text": string | null}'
_FIELD_NOTES = {
    "full_name": "holder's full name",
    "licence_number": "driving licence number",
    "date_of_birth": "value as YYYY-MM-DD",
    "date_of_issue": "value as YYYY-MM-DD",
    "date_of_expiry": "value as YYYY-MM-DD (valid till / expiry)",
    "address": "full address, may span several lines",
    "vehicle_classes": "authorised vehicle classes, comma-joined if multiple",
    "issuing_authority": "issuing authority / office",
}
EXTRACTION_USER_PROMPT = (
    "Extract the driving licence fields from this image. Respond with a JSON object of exactly "
    "this shape:\n{\n"
    + "".join(f'  "{name}": {_FIELD},  // {_FIELD_NOTES[name]}\n' for name in CORE_FIELDS)
    + f'  "other_fields": {{ "<descriptive_snake_case_key>": {_FIELD}, ... }}\n}}'
)
JSON_RETRY_PROMPT = "Your previous response was not valid JSON. Return only the JSON object."


class ExtractionProvider(Protocol):
    async def extract(self, image_bytes: bytes, media_type: str) -> LicenceData:
        """Extract licence fields. bbox is left None; it is filled later by the merge step."""
        ...


class ProviderError(Exception):
    """Human-readable failure from an extraction/chat provider (safe to show to users)."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class InvalidJSONError(ValueError):
    pass


def _as_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v if x is not None)
    s = str(v).strip()
    return s or None


def _field(raw) -> FieldValue:
    if isinstance(raw, dict):
        value, source = _as_text(raw.get("value")), _as_text(raw.get("source_text"))
    else:  # bare value instead of {value, source_text}
        value, source = _as_text(raw), None
    if value is None:
        source = None
    return FieldValue(value=value, source_text=source, confidence="review", bbox=None)


def parse_licence_json(text: str | None) -> LicenceData:
    """Parse a model reply into LicenceData. Raises InvalidJSONError if no JSON object is found."""
    if not text:
        raise InvalidJSONError("empty response")
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise InvalidJSONError("no JSON object in response")
    try:
        # strict=False tolerates raw newlines inside strings (multi-line addresses).
        obj = json.loads(cleaned[start : end + 1], strict=False)
    except json.JSONDecodeError as e:
        raise InvalidJSONError(str(e)) from e
    if not isinstance(obj, dict):
        raise InvalidJSONError("top-level JSON is not an object")

    other_raw = obj.get("other_fields")
    other = {}
    if isinstance(other_raw, dict):
        for key, raw in other_raw.items():
            key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if key and key not in CORE_FIELDS:
                other[key] = _field(raw)
    return LicenceData(**{name: _field(obj.get(name)) for name in CORE_FIELDS}, other_fields=other)


def llm_model() -> str:
    """Extraction model: LLM_MODEL, else the default chosen by scripts/compare_models.py."""
    return (os.getenv("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def chat_model() -> str:
    """Chat model: LLM_CHAT_MODEL if set, else the extraction model."""
    return (os.getenv("LLM_CHAT_MODEL") or "").strip() or llm_model()


def get_provider(model: str | None = None) -> ExtractionProvider:
    """The extraction provider for `model` (default: LLM_MODEL).

    Routes only ever see the ExtractionProvider Protocol. Taking an explicit model lets
    scripts/compare_models.py run two models through identical code.
    """
    model = (model or "").strip() or llm_model()
    if model.startswith("ollama/"):
        raise ProviderError("Ollama models are not supported yet. Set LLM_MODEL to an OpenRouter model.")
    # Imported here: provider modules import this module for the shared prompt and parser.
    from app.services.providers.openrouter import OpenRouterProvider

    return OpenRouterProvider(model)
