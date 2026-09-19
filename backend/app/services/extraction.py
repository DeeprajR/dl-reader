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


_DATE_IN_TEXT = re.compile(
    r"\d{4}[-/. ]\d{1,2}[-/. ]\d{1,2}"
    r"|\d{1,2} ?[-/.] ?\d{1,2} ?[-/.] ?(?:\d{4}|\d{2})"
    r"|\d{1,2}(?:st|nd|rd|th)?[ ./-]?[A-Za-z]{3,9}\.?[ ./-]?(?:\d{4}|\d{2})"
    r"|[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}"
)


def find_date(text: str | None) -> str | None:
    """First date found inside `text` (e.g. "DOI: 16-06-2019" -> "2019-06-16"), or None."""
    if not text:
        return None
    if iso := normalize_date(text):
        return iso
    for m in _DATE_IN_TEXT.finditer(text):
        if iso := normalize_date(m.group()):
            return iso
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


MIN_PART_CHARS = 3


def match_bbox_parts(target: str | None, words: list[dict]) -> Box | None:
    """Fallback for values whose parts are not consecutive in OCR reading order.

    A table column such as "LMV\\nMCWG" is read row by row by Tesseract, so the whole value
    never forms one window. Each part (split on lines, commas, semicolons) is matched on its
    own and the boxes of the parts that were found are unioned. Parts shorter than
    MIN_PART_CHARS are ignored: they would match almost anywhere.
    """
    parts = [p for p in re.split(r"[\n,;]+", target or "") if len(normalize(p)) >= MIN_PART_CHARS]
    if len(parts) < 2:
        return None
    boxes = [box for box in (match_bbox(p, words) for p in parts) if box]
    if not boxes:
        return None
    left, top = min(b.x for b in boxes), min(b.y for b in boxes)
    right, bottom = max(b.x + b.w for b in boxes), max(b.y + b.h for b in boxes)
    return Box(x=left, y=top, w=right - left, h=bottom - top)


# --- per-class validity and date order ------------------------------------------------------

# other_fields written per vehicle class, e.g. "lmv_date_of_issue", "mcwg_valid_till".
CLASS_DATE_KEY = re.compile(r"(?P<cls>[a-z0-9_]+?)_(?P<kind>date_of_issue|valid_till)")


def class_date_key(name: str) -> re.Match | None:
    return CLASS_DATE_KEY.fullmatch(name.removeprefix("other_fields."))


def _class_row(field: FieldValue, words: list[dict]) -> tuple[Box | None, bool]:
    """Locate a per-class date by its class code (the first word of its row source_text).

    Returns the box of that class's row and whether the value's date is printed on that row.
    The same dates often repeat on every row, so matching the row text alone can land on the
    wrong class; anchoring on the code avoids that, and an unreadable code gives no box.
    """
    parts = (field.source_text or "").split()
    if not parts or find_date(parts[0]) or not words:
        return None, False
    anchor = match_bbox(parts[0], words)
    if anchor is None:
        return None, False
    row = [w for w in words if anchor.y <= w["top"] + w["height"] / 2 <= anchor.y + anchor.h]
    confirmed = any(find_date(w["text"]) == field.value for w in row)
    return match_bbox(field.source_text, row) or anchor, confirmed


def _as_date(field: FieldValue | None) -> date | None:
    try:
        return date.fromisoformat(field.value) if field and field.value else None
    except ValueError:
        return None


def check_date_order(data: LicenceData) -> list[str]:
    """Flag impossible dates for review: birth < issue < expiry (licence and per class), and no
    birth or issue date in the future. Catches swapped dates whatever caused them."""
    warnings: list[str] = []
    today = date.today()

    def flag(message: str, *fields: FieldValue) -> None:
        warnings.append(message)
        for field in fields:
            field.confidence = "review"

    dob, doi, doe = data.date_of_birth, data.date_of_issue, data.date_of_expiry
    birth, issue, expiry = _as_date(dob), _as_date(doi), _as_date(doe)
    if birth and birth > today:
        flag(f"Date of birth ({birth}) is in the future.", dob)
    if issue and issue > today:
        flag(f"Date of issue ({issue}) is in the future.", doi)
    if issue and expiry and issue >= expiry:
        flag(f"Date of issue ({issue}) is not before the expiry date ({expiry}). Were they swapped?", doi, doe)
    if birth and issue and birth >= issue:
        flag(f"Date of birth ({birth}) is not before the date of issue ({issue}).", dob, doi)

    per_class: dict[str, dict[str, FieldValue]] = {}
    for key, field in data.other_fields.items():
        if m := class_date_key(key):
            per_class.setdefault(m["cls"], {})[m["kind"]] = field
    for cls, pair in per_class.items():
        label = cls.upper().replace("_", " ")
        cls_issue, cls_till = pair.get("date_of_issue"), pair.get("valid_till")
        start, end = _as_date(cls_issue), _as_date(cls_till)
        if start and start > today:
            flag(f"{label} date of issue ({start}) is in the future.", cls_issue)
        if start and end and start >= end:
            flag(
                f"{label} date of issue ({start}) is not before its valid-till date ({end}). Were they swapped?",
                cls_issue,
                cls_till,
            )
        if birth and start and birth >= start:
            flag(f"Date of birth ({birth}) is not before the {label} date of issue ({start}).", dob, cls_issue)
    return warnings


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
        if value and (name in DATE_FIELDS or class_date_key(name)):
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

    Per-class dates are anchored on their class's table row, and impossible date orders are
    flagged. Returns a new LicenceData plus warnings. Null-value fields are always "review".
    """
    data = data.model_copy(deep=True)
    warnings: list[str] = []
    low_ocr = len(ocr_text.strip()) < MIN_OCR_CHARS
    if low_ocr:
        warnings.append(LOW_OCR_WARNING)

    words = words or []
    for name, field in iter_fields(data):
        per_class = class_date_key(name) is not None
        is_date = name in DATE_FIELDS or per_class
        field.page = page
        field.confidence = "review"
        field.bbox = None

        if is_date and field.value is not None:
            # source_text may carry the printed label ("DOI: 16-06-2019"); a per-class row
            # holds two dates, so only the value itself is trusted there.
            iso = find_date(field.value) or (None if per_class else find_date(field.source_text))
            if iso:
                field.value = iso
            else:
                warnings.append(f"{name}: could not normalise the date '{field.value}'")

        if field.value is None:
            continue
        if per_class:
            field.bbox, on_its_row = _class_row(field, words)
            if on_its_row and not low_ocr:
                field.confidence = "high"
            continue
        anchor = field.source_text or field.value
        field.bbox = match_bbox(anchor, words) or match_bbox_parts(anchor, words)
        if low_ocr:
            continue
        if text_matches_ocr(anchor, ocr_text, is_date=is_date):
            field.confidence = "high"

    warnings += check_date_order(data)
    return data, warnings
