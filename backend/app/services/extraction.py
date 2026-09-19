"""Merge LLM extraction with OCR: date normalisation, confidence flags and bbox matching."""

import re
import string
from datetime import date
from difflib import SequenceMatcher

from app.schemas import CORE_FIELDS, DATE_FIELDS, Box, FieldValue, LicenceData

TEXT_MATCH_RATIO = 0.85
BBOX_MATCH_RATIO = 0.8
MIN_OCR_CHARS = 20
LOW_OCR_WARNING = "OCR yielded little text; confidence flags unreliable"

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "“”‘’—–·•|"})
_MONTHS = (
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
)  # fmt: skip
_SEP = r"(?: ?[-/.] ?| )"


# --- normalisation --------------------------------------------------------------------------


def normalize(s: str | None) -> str:
    """Uppercase, strip punctuation, collapse whitespace."""
    if not s:
        return ""
    return " ".join(s.upper().translate(_PUNCT_TABLE).split())


def digits_only(s: str | None) -> str:
    return re.sub(r"\D", "", s or "")


def _month(name: str) -> int | None:
    if len(name) < 3:
        return None
    for i, full in enumerate(_MONTHS, start=1):
        if full.startswith(name):
            return i
    return None


def _year(y: str) -> int:
    if len(y) == 4:
        return int(y)
    # Two-digit year: pick the century that keeps the date within ~20 years of today.
    yy = int(y)
    return 2000 + yy if 2000 + yy <= date.today().year + 20 else 1900 + yy


def _build(y: int, m: int | None, d: int) -> str | None:
    if m is None:
        return None
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def normalize_date(s: str | None) -> str | None:
    """Parse a printed date into YYYY-MM-DD (day-first for numeric dates). Unparseable -> None."""
    if not s:
        return None
    t = " ".join(s.upper().replace(",", " ").split())

    if m := re.fullmatch(rf"(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})", t):
        return _build(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(rf"(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}}|\d{{2}})", t):
        return _build(_year(m[3]), int(m[2]), int(m[1]))
    if m := re.fullmatch(r"(\d{1,2})(?:ST|ND|RD|TH)?[ ./-]?([A-Z]{3,9})\.?[ ./-]?(\d{4}|\d{2})", t):
        return _build(_year(m[3]), _month(m[2]), int(m[1]))
    if m := re.fullmatch(r"([A-Z]{3,9})\.? (\d{1,2})(?:ST|ND|RD|TH)? (\d{4})", t):
        return _build(int(m[3]), _month(m[1]), int(m[2]))
    return None


# --- fuzzy matching -------------------------------------------------------------------------


def _best_window(target: str, units: list[str], n: int) -> tuple[float, int, int]:
    """Best (score, start, end) over windows of `n` ±1 consecutive units, by difflib ratio.

    Ties keep the earliest window (reading order).
    """
    matcher = SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(target)  # seq2 is cached; only the window changes
    best = (0.0, 0, 0)
    for size in (n, n - 1, n + 1):
        if size < 1 or size > len(units):
            continue
        for start in range(len(units) - size + 1):
            matcher.set_seq1(" ".join(units[start : start + size]))
            if matcher.real_quick_ratio() <= best[0] or matcher.quick_ratio() <= best[0]:
                continue  # cannot beat the current best
            score = matcher.ratio()
            if score > best[0]:
                best = (score, start, start + size)
                if score == 1.0:
                    return best
    return best


def text_matches_ocr(target: str | None, ocr_text: str, *, is_date: bool = False) -> bool:
    """True if `target` appears in the OCR text, exactly or with ratio >= TEXT_MATCH_RATIO."""
    norm_target = normalize(target)
    if not norm_target:
        return False
    norm_ocr = normalize(ocr_text)
    if f" {norm_target} " in f" {norm_ocr} ":
        return True
    if is_date:
        target_digits = digits_only(target)
        if len(target_digits) >= 6 and target_digits in digits_only(ocr_text):
            return True
    tokens = norm_target.split()
    score, _, _ = _best_window(norm_target, norm_ocr.split(), len(tokens))
    return score >= TEXT_MATCH_RATIO


def match_bbox(target: str | None, words: list[dict]) -> Box | None:
    """Locate `target` in the Tesseract word list; the union box of the best window, or None.

    Windows of consecutive words (target word count ±1) are scored by the difflib ratio of
    their joined normalised text against the normalised target; below BBOX_MATCH_RATIO there
    is no box. A window may cross lines, so a multi-line value such as an address gets one
    box covering all of its lines. Pure function; None simply means "show the snippet only".
    """
    norm_target = normalize(target)
    if not norm_target or not words:
        return None
    # Punctuation-only words (":", "-", "|") normalise to nothing: drop them so they neither
    # count as words nor break a window, on both sides.
    n = sum(1 for part in (target or "").split() if normalize(part))
    kept = [(normalize(w["text"]), w) for w in words]
    kept = [(text, w) for text, w in kept if text]
    if not kept:
        return None

    score, start, end = _best_window(norm_target, [text for text, _ in kept], n)
    if score < BBOX_MATCH_RATIO:
        return None
    window = [w for _, w in kept[start:end]]
    left = min(w["left"] for w in window)
    top = min(w["top"] for w in window)
    right = max(w["left"] + w["width"] for w in window)
    bottom = max(w["top"] + w["height"] for w in window)
    return Box(x=left, y=top, w=right - left, h=bottom - top)


# --- user edits -----------------------------------------------------------------------------

MAX_VALUE_CHARS = 1000
_OTHER_KEY = re.compile(r"[a-z0-9_]{1,64}")


def clean_user_data(data: LicenceData) -> LicenceData:
    """Validate and tidy user-edited data. Raises ValueError with a user-facing message."""
    data = data.model_copy(deep=True)
    for key in data.other_fields:
        if not _OTHER_KEY.fullmatch(key):
            raise ValueError(f"Invalid field name '{key}'")
    for name, field in iter_fields(data):
        value = (field.value or "").strip() or None
        if value and len(value) > MAX_VALUE_CHARS:
            raise ValueError(f"{name} is longer than {MAX_VALUE_CHARS} characters")
        if value and name in DATE_FIELDS:
            iso = normalize_date(value)
            if iso is None:
                raise ValueError(f"{name} must be a valid date in YYYY-MM-DD format")
            value = iso
        field.value = value
    return data


# --- merge ----------------------------------------------------------------------------------


def iter_fields(data: LicenceData) -> list[tuple[str, FieldValue]]:
    """(name, field) for every field; other_fields are named "other_fields.<key>"."""
    core = [(name, getattr(data, name)) for name in CORE_FIELDS]
    return core + [(f"other_fields.{k}", v) for k, v in data.other_fields.items()]


def merge(
    data: LicenceData, ocr_text: str, words: list[dict] | None = None, page: int = 1
) -> tuple[LicenceData, list[str]]:
    """Cross-check every field against OCR: confidence from the text, bbox from word boxes.

    Returns a new LicenceData plus warnings. Null-value fields are always "review" with no box.
    """
    data = data.model_copy(deep=True)
    warnings: list[str] = []
    low_ocr = len(ocr_text.strip()) < MIN_OCR_CHARS
    if low_ocr:
        warnings.append(LOW_OCR_WARNING)

    for name, field in iter_fields(data):
        is_date = name in DATE_FIELDS
        field.page = page
        field.confidence = "review"
        field.bbox = None

        if is_date and field.value is not None:
            iso = normalize_date(field.value) or normalize_date(field.source_text)
            if iso:
                field.value = iso
            else:
                warnings.append(f"{name}: could not normalise the date '{field.value}'")

        if field.value is None:
            continue
        field.bbox = match_bbox(field.source_text or field.value, words or [])
        if low_ocr:
            continue
        if text_matches_ocr(field.source_text or field.value, ocr_text, is_date=is_date):
            field.confidence = "high"

    return data, warnings
