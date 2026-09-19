"""Extraction provider Protocol, factory, and the prompt/parsing shared by all providers."""

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Protocol

from app.schemas import CORE_FIELDS, FieldValue, LicenceData

logger = logging.getLogger(__name__)

# Used when LLM_MODEL is not set. Chosen with scripts/compare_models.py (see the README).
DEFAULT_MODEL = "google/gemini-3.8-flash"
# A model name that starts with this runs locally through Ollama, e.g. "ollama/qwen2.5vl:3b".
OLLAMA_PREFIX = "ollama/"
MAX_ATTEMPTS = 2  # one retry on failure or invalid JSON

# The LLM's instructions for reading a licence, word for word from the specification. Rule 1
# (copy the printed text) and rule 2 (never guess) are what make the OCR cross-check possible.
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

# The user prompt shows the LLM the exact JSON shape to return. It is built from CORE_FIELDS, so
# the prompt and the data model cannot drift apart.
_FIELD = '{"value": string | null, "source_text": string | null}'
# A hint per field: what it is, and the labels it is usually printed under.
_FIELD_NOTES = {
    "full_name": "holder's full name",
    "licence_number": "driving licence number",
    "date_of_birth": "value as YYYY-MM-DD; printed as e.g. DOB / Date of Birth",
    "date_of_issue": "value as YYYY-MM-DD; printed as e.g. DOI / Date of Issue / Issued on",
    "date_of_expiry": "value as YYYY-MM-DD; printed as e.g. Valid Till / Validity / Valid Upto / Expiry",
    "address": "full address, may span several lines",
    "vehicle_classes": "authorised vehicle classes, comma-joined if multiple",
    "issuing_authority": "issuing authority / office",
}
EXTRACTION_USER_PROMPT = (
    "Extract the driving licence fields from this image. Respond with a JSON object of exactly "
    "this shape:\n{\n"
    + "".join(f'  "{name}": {_FIELD},  // {_FIELD_NOTES[name]}\n' for name in CORE_FIELDS)
    + f'  "other_fields": {{ "<descriptive_snake_case_key>": {_FIELD}, ... }}\n}}'
    # The printed label shows which date is which, to the reviewer and in the chat excerpts.
    + "\nFor date_of_birth, date_of_issue and date_of_expiry, source_text is the printed label "
    'together with the date, exactly as printed (e.g. "DOI: 01-02-2020", "Valid Till: '
    '31-01-2040").'
    # Per-class validity has no core field; without this hint models drop the dates.
    + "\nIf the licence prints validity per vehicle class (for example a table of class, issue "
    "date and valid-till date), add two other_fields for every class: "
    '"<class>_date_of_issue" and "<class>_valid_till", with the class code in snake_case '
    "(e.g. lmv_date_of_issue, mcwg_valid_till). Value: that date as YYYY-MM-DD. source_text: "
    'that class\'s table row exactly as printed (e.g. "LMV 01-02-2020 31-01-2040"). Omit them '
    "if no such table is printed."
)
# Sent after a reply that was not valid JSON, on the one retry.
JSON_RETRY_PROMPT = "Your previous response was not valid JSON. Return only the JSON object."


class ExtractionProvider(Protocol):
    """What every LLM provider must offer. The routes only know this interface, so a new provider
    can be added without touching them.
    """

    async def extract(self, image_bytes: bytes, media_type: str) -> LicenceData:
        """Extract licence fields. bbox is left None; it is filled later by the merge step."""
        ...


class ProviderError(Exception):
    """Human-readable failure from an extraction/chat provider (safe to show to users)."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        # True for failures that may pass on their own (timeout, busy server), False for the rest (bad key).
        self.retryable = retryable


class InvalidJSONError(ValueError):
    """The model's reply could not be parsed as the expected JSON object."""

    pass


def _as_text(v) -> str | None:
    """Any JSON value as clean text, or None. Models sometimes return a list or a number where text is expected."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v if x is not None)
    s = str(v).strip()
    return s or None


def _field(raw) -> FieldValue:
    """One field of the reply as a FieldValue. It always starts as "review" with no box: those are set by the merge."""
    if isinstance(raw, dict):
        value, source = _as_text(raw.get("value")), _as_text(raw.get("source_text"))
    else:  # bare value instead of {value, source_text}
        value, source = _as_text(raw), None
    # A source text without a value is meaningless, so it is dropped.
    if value is None:
        source = None
    return FieldValue(value=value, source_text=source, confidence="review", bbox=None)


def parse_licence_json(text: str | None) -> LicenceData:
    """Parse a model reply into LicenceData. Raises InvalidJSONError if no JSON object is found."""
    if not text:
        raise InvalidJSONError("empty response")
    # Models often wrap JSON in markdown fences or add a sentence around it. Keep the outermost { ... } only.
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

    # "other_fields" keys are made safe: lowercase snake_case, and never the name of a core field.
    other_raw = obj.get("other_fields")
    other = {}
    if isinstance(other_raw, dict):
        for key, raw in other_raw.items():
            key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if key and key not in CORE_FIELDS:
                other[key] = _field(raw)
    # A core field missing from the reply becomes an empty field instead of an error.
    return LicenceData(**{name: _field(obj.get(name)) for name in CORE_FIELDS}, other_fields=other)


async def extract_with_retry(
    complete: Callable[[list[dict]], Awaitable[str]], messages: list[dict], model: str
) -> LicenceData:
    """Shared by every provider: call, parse, and retry once on a transient failure or invalid
    JSON (appending the JSON_RETRY_PROMPT after the bad reply). Non-retryable errors raise."""
    last_error: ProviderError | None = None
    # Two kinds of failure are retried once: the call itself failing, and a reply that is not valid JSON.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            reply = await complete(messages)
        except ProviderError as e:
            if not e.retryable:
                raise
            logger.warning("Extraction attempt %d with %s failed: %s", attempt, model, e)
            last_error = e
            continue
        try:
            return parse_licence_json(reply)
        except InvalidJSONError as e:
            logger.warning("Attempt %d with %s returned invalid JSON: %s", attempt, model, e)
            last_error = ProviderError(f"The model {model} did not return valid JSON.")
            if reply.strip():  # an empty reply is simply retried as-is
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": JSON_RETRY_PROMPT},
                ]
    # Both attempts failed: report the last error.
    assert last_error is not None
    raise last_error


def is_local(model: str) -> bool:
    """True when the model runs on this machine through Ollama."""
    return model.startswith(OLLAMA_PREFIX)


def llm_model() -> str:
    """Extraction model: LLM_MODEL, else the default chosen by scripts/compare_models.py."""
    return (os.getenv("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def chat_model() -> str:
    """Chat model: LLM_CHAT_MODEL if set, else the extraction model."""
    return (os.getenv("LLM_CHAT_MODEL") or "").strip() or llm_model()


def get_provider(model: str | None = None) -> ExtractionProvider:
    """The extraction provider for `model` (default: LLM_MODEL): "ollama/<name>" runs locally
    through Ollama, anything else goes to OpenRouter.

    Routes only ever see the ExtractionProvider Protocol. Taking an explicit model lets
    scripts/compare_models.py run two models through identical code.
    """
    model = (model or "").strip() or llm_model()
    # Imported here: provider modules import this module for the shared prompt and parser.
    if is_local(model):
        from app.services.providers.ollama import OllamaProvider

        return OllamaProvider(model.removeprefix(OLLAMA_PREFIX))
    from app.services.providers.openrouter import OpenRouterProvider

    return OpenRouterProvider(model)
