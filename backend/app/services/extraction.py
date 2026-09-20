"""Cross-check the AI model's answer against OCR.

The AI model returns each field with the text it copied from the card (`source_text`). This
module decides, field by field:

  * confidence - "high" when OCR also read that text, otherwise "review" ("Please verify")
  * bbox       - where the text is on the image, from OCR's word positions, for the highlight
  * score      - how sure OCR was about the printed words the value was found in (0-100)
  * dates      - normalised to YYYY-MM-DD, and checked for an impossible order

Everything here is pure logic on text and numbers: no files, no network, no AI calls.
"""

import re
import string
from datetime import date
from difflib import SequenceMatcher

from app.schemas import CORE_FIELDS, DATE_FIELDS, Box, FieldValue, LicenceData

# How similar (0 to 1) the AI's text and the OCR text must be to count as "the same text".
# OCR misreads a character now and then, so an exact match would reject too much.
TEXT_MATCH_RATIO = 0.85
# The same idea for finding the text's position. Slightly looser, because a wrong highlight is
# harmless while a wrong "confirmed" is not.
BBOX_MATCH_RATIO = 0.8
# The shortest part of a multi-part value ("LMV, MCWG") that is matched on its own. Shorter
# parts, such as "A", would match almost anywhere.
MIN_PART_CHARS = 3
# With less OCR text than this, OCR has effectively failed and nothing can be confirmed.
MIN_OCR_CHARS = 20
LOW_OCR_WARNING = "OCR yielded little text; confidence flags unreliable"

# Translation table that turns every punctuation mark into a space (used by `normalize`).
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "“”‘’—–·•|"})
_MONTHS = (
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
)  # fmt: skip
# The separator between the parts of a numeric date: "-", "/", "." (with optional spaces) or
# a single space.
_SEP = r"(?: ?[-/.] ?| )"


# --- normalisation --------------------------------------------------------------------------
# Text from the AI and text from OCR differ in case, punctuation and spacing. Both are reduced
# to the same plain form before they are compared.


def normalize(s: str | None) -> str:
    """Uppercase, punctuation removed, whitespace collapsed: "D.O.B :  12-08" -> "D O B 12 08"."""
    if not s:
        return ""
    return " ".join(s.upper().translate(_PUNCT_TABLE).split())


def digits_only(s: str | None) -> str:
    """Only the digits of `s`: "12-08-1990" -> "12081990". Used to compare dates."""
    return re.sub(r"\D", "", s or "")


def _month(name: str) -> int | None:
    """Month number of a full or shortened English month name ("SEP", "SEPT", "SEPTEMBER" -> 9)."""
    # Fewer than three letters is ambiguous ("JU" could be June or July).
    if len(name) < 3:
        return None
    for i, full in enumerate(_MONTHS, start=1):
        if full.startswith(name):
            return i
    return None


def _year(y: str) -> int:
    """A four-digit year from "2034" or from a two-digit year such as "34"."""
    if len(y) == 4:
        return int(y)
    # Two-digit year: pick the century that keeps the date within ~20 years of today.
    # Expiry dates can lie in the future, birth dates cannot lie far in it.
    yy = int(y)
    return 2000 + yy if 2000 + yy <= date.today().year + 20 else 1900 + yy


def _build(y: int, m: int | None, d: int) -> str | None:
    """YYYY-MM-DD for a real calendar date, or None (unknown month, 31 February, ...)."""
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
    # Uppercase, commas removed, single spaces: "16 jun, 2019" -> "16 JUN 2019".
    t = " ".join(s.upper().replace(",", " ").split())

    # The whole string must be a date. The formats are tried from the least to the most
    # ambiguous.
    # 1. Year first: 2019-06-16
    if m := re.fullmatch(rf"(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})", t):
        return _build(int(m[1]), int(m[2]), int(m[3]))
    # 2. Day first, as printed on Indian licences: 16-06-2019 or 16/06/19
    if m := re.fullmatch(rf"(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}}|\d{{2}})", t):
        return _build(_year(m[3]), int(m[2]), int(m[1]))
    # 3. Day, month name, year: 16 JUN 2019, 16th June 2019, 16-JUN-19
    if m := re.fullmatch(r"(\d{1,2})(?:ST|ND|RD|TH)?[ ./-]?([A-Z]{3,9})\.?[ ./-]?(\d{4}|\d{2})", t):
        return _build(_year(m[3]), _month(m[2]), int(m[1]))
    # 4. Month name first: JUNE 16 2019
    if m := re.fullmatch(r"([A-Z]{3,9})\.? (\d{1,2})(?:ST|ND|RD|TH)? (\d{4})", t):
        return _build(int(m[3]), _month(m[1]), int(m[2]))
    return None


# The same four date formats, but able to match in the middle of a longer text.
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
    # The text may be nothing but a date.
    if iso := normalize_date(text):
        return iso
    # Otherwise look for a date inside it, skipping look-alikes that are not real dates.
    for m in _DATE_IN_TEXT.finditer(text):
        if iso := normalize_date(m.group()):
            return iso
    return None


# --- fuzzy matching -------------------------------------------------------------------------
# "Is this text printed on the card?" and "where?". Both slide a window of words over the
# OCR output and score each window against the target with difflib's similarity ratio.


def _best_window(target: str, units: list[str], n: int) -> tuple[float, int, int]:
    """Best (score, start, end) over windows of `n` ±1 consecutive units, by difflib ratio.

    Ties keep the earliest window (reading order).
    """
    matcher = SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(target)  # seq2 is cached; only the window changes
    best = (0.0, 0, 0)
    # OCR may split one word in two or join two words, so windows one word shorter and one
    # word longer than the target are tried as well.
    for size in (n, n - 1, n + 1):
        if size < 1 or size > len(units):
            continue
        for start in range(len(units) - size + 1):
            matcher.set_seq1(" ".join(units[start : start + size]))
            # The two quick ratios are cheap upper bounds of the real ratio. Skipping the
            # windows that cannot win keeps this fast on cards with a lot of text.
            if matcher.real_quick_ratio() <= best[0] or matcher.quick_ratio() <= best[0]:
                continue  # cannot beat the current best
            score = matcher.ratio()
            if score > best[0]:
                best = (score, start, start + size)
                if score == 1.0:
                    return best  # a perfect match cannot be beaten
    return best


def text_matches_ocr(target: str | None, ocr_text: str, *, is_date: bool = False) -> bool:
    """True if `target` appears in the OCR text, exactly or with ratio >= TEXT_MATCH_RATIO."""
    norm_target = normalize(target)
    if not norm_target:
        return False
    norm_ocr = normalize(ocr_text)
    # 1. Exact match on whole words. The padding spaces stop "MH12" matching inside "XMH123".
    if f" {norm_target} " in f" {norm_ocr} ":
        return True
    # 2. Dates: compare the digits only, because OCR often misreads or drops the separators
    #    ("16-06-2019" read as "16.06 2019"). Six digits is the shortest full date.
    if is_date:
        target_digits = digits_only(target)
        if len(target_digits) >= 6 and target_digits in digits_only(ocr_text):
            return True
    # 3. Fuzzy match, which tolerates a misread character or two.
    tokens = norm_target.split()
    score, _, _ = _best_window(norm_target, norm_ocr.split(), len(tokens))
    return score >= TEXT_MATCH_RATIO


def parts_match_ocr(target: str | None, ocr_text: str) -> bool:
    """True if every part of a multi-part value is printed as whole words in the OCR text.

    A table column such as "LMV\nMCWG" is never contiguous in OCR reading order. Parts
    shorter than MIN_PART_CHARS cannot be checked safely, so their presence means False.
    """
    # Split on line breaks, commas and semicolons, and drop parts that are only punctuation.
    parts = [normalize(p) for p in re.split(r"[\n,;]+", target or "") if normalize(p)]
    if len(parts) < 2 or any(len(p) < MIN_PART_CHARS for p in parts):
        return False
    padded = f" {normalize(ocr_text)} "
    return all(f" {part} " in padded for part in parts)


def match_words(target: str | None, words: list[dict]) -> list[dict]:
    """Locate `target` in the Tesseract word list: the words of the best window, or [].

    Windows of consecutive words (target word count ±1) are scored by the difflib ratio of
    their joined normalised text against the normalised target; below BBOX_MATCH_RATIO nothing
    is found. A window may cross lines, so a multi-line value such as an address is found as
    one run of words covering all of its lines. Pure function.
    """
    norm_target = normalize(target)
    if not norm_target or not words:
        return []
    # Punctuation-only words (":", "-", "|") normalise to nothing: drop them so they neither
    # count as words nor break a window, on both sides.
    n = sum(1 for part in (target or "").split() if normalize(part))
    kept = [(normalize(w["text"]), w) for w in words]
    kept = [(text, w) for text, w in kept if text]
    if not kept:
        return []

    score, start, end = _best_window(norm_target, [text for text, _ in kept], n)
    if score < BBOX_MATCH_RATIO:
        return []
    return [w for _, w in kept[start:end]]


def match_bbox(target: str | None, words: list[dict]) -> Box | None:
    """Where `target` is on the image: the smallest rectangle around the words it was matched
    to, or None ("show the source text only")."""
    found = match_words(target, words)
    return _box_of(found) if found else None


def match_words_parts(target: str | None, words: list[dict]) -> list[dict]:
    """Fallback for values whose parts are not consecutive in OCR reading order.

    A table column such as "LMV\\nMCWG" is read row by row by Tesseract, so the whole value
    never forms one window. Each part (split on lines, commas, semicolons) is matched on its
    own and the words of the parts that were found are returned together. Parts shorter than
    MIN_PART_CHARS are ignored: they would match almost anywhere.
    """
    parts = [p for p in re.split(r"[\n,;]+", target or "") if len(normalize(p)) >= MIN_PART_CHARS]
    if len(parts) < 2:
        return []
    # Find each part on its own. A word that two parts share is kept once.
    found: dict[int, dict] = {}
    for part in parts:
        for w in match_words(part, words):
            found[id(w)] = w
    return list(found.values())


def match_bbox_parts(target: str | None, words: list[dict]) -> Box | None:
    """One rectangle around every part of a multi-part value that was found, or None."""
    found = match_words_parts(target, words)
    return _box_of(found) if found else None


def confidence_score(words: list[dict]) -> int | None:
    """How sure OCR was about `words`: Tesseract's average certainty, from 0 to 100.

    Tesseract rates every word it reads. A low score means the print was hard to read (blurred,
    tiny, on a busy background), so the value deserves a closer look even when it matched.
    None when there are no words, i.e. the value was not located on the image.
    """
    scores = [w["conf"] for w in words if w.get("conf") is not None]
    return round(sum(scores) / len(scores)) if scores else None


# --- per-class validity and date order ------------------------------------------------------
# A licence may list several vehicle classes in a table, each with its own issue and
# valid-till date. The same dates often repeat on every row, so a per-class date cannot be
# located by its text alone: it has to be found on the row of its class.

# other_fields written per vehicle class, e.g. "lmv_date_of_issue", "mcwg_valid_till".
CLASS_DATE_KEY = re.compile(r"(?P<cls>[a-z0-9_]+?)_(?P<kind>date_of_issue|valid_till)")


def class_date_key(name: str) -> re.Match | None:
    """Match for a per-class date field name, with groups `cls` ("lmv") and `kind`; else None."""
    return CLASS_DATE_KEY.fullmatch(name.removeprefix("other_fields."))


def _class_label(source_text: str | None) -> list[str]:
    """Normalised tokens of a row's class cell: everything before its first date."""
    tokens: list[str] = []
    for part in (source_text or "").split():
        if find_date(part):
            break
        tokens += normalize(part).split()
    return tokens


def _box_of(words: list[dict]) -> Box:
    """The smallest rectangle that contains all of `words`."""
    left, top = min(w["left"] for w in words), min(w["top"] for w in words)
    right = max(w["left"] + w["width"] for w in words)
    bottom = max(w["top"] + w["height"] for w in words)
    return Box(x=left, y=top, w=right - left, h=bottom - top)


def _visual_rows(words: list[dict]) -> list[list[dict]]:
    """Words grouped by vertical position into rows, each sorted left to right.

    Tesseract's own line ids cannot be trusted for tables: once rulings are erased it may
    split one table row into separate column blocks.
    """
    if not words:
        return []
    # Two words are on the same row when their vertical centres are within 60% of the
    # typical (median) word height of each other.
    heights = sorted(w["height"] for w in words)
    tolerance = 0.6 * heights[len(heights) // 2]
    rows: list[list[dict]] = []
    centre = None
    # Walk the words from top to bottom. Each one joins the current row or starts a new one.
    for w in sorted(words, key=lambda w: w["top"] + w["height"] / 2):
        mid = w["top"] + w["height"] / 2
        if rows and abs(mid - centre) <= tolerance:
            rows[-1].append(w)
            # Keep `centre` as the running average of the row, so a slightly tilted scan
            # does not drift out of tolerance along the row.
            centre += (mid - centre) / len(rows[-1])
        else:
            rows.append([w])
            centre = mid
    return [sorted(row, key=lambda w: w["left"]) for row in rows]


def _class_row(field: FieldValue, words: list[dict]) -> tuple[list[dict], bool]:
    """Locate a per-class date by the full class label of its row source_text ("LMV TR").

    In each visual row, a class cell is the run of non-date words just before a date; the
    label must match such a cell exactly (as its trailing words), and the class's dates are
    the date words that follow. Fuzzy or first-word matching confuses classes such as LMV /
    LMV NT / LMV TR or MCWG / MCWOG, and the same dates often repeat on every row, so a label
    found on no row or several gives nothing rather than the wrong row. Returns the words of
    the label and its dates, and whether the value's date is among them.
    """
    label = _class_label(field.source_text)
    if not label:
        return [], False
    # Every (label words, date words) pair found on the card for this label.
    matches = []
    for row in _visual_rows(words):
        # `cell` collects the non-date words seen since the last date: the current class cell.
        cell: list[tuple[str, dict]] = []
        for i, w in enumerate(row):
            if not find_date(w["text"]):
                cell += [(token, w) for token in normalize(w["text"]).split()]
                continue
            # `w` is a date, so the class cell just ended. Is it exactly our label?
            if len(cell) >= len(label) and [t for t, _ in cell[-len(label):]] == label:
                # Collect the dates that follow, up to the next piece of ordinary text.
                dates = []
                for nxt in row[i:]:
                    if find_date(nxt["text"]):
                        dates.append(nxt)
                    elif normalize(nxt["text"]):
                        break  # the next class cell (or other text) begins
                # One OCR word can hold several label tokens ("LMV-TR"), so the words are
                # de-duplicated by identity while their order is kept.
                label_words = list(dict.fromkeys(id(w) for _, w in cell[-len(label):]))
                matches.append(([w for w in row if id(w) in label_words], dates))
            cell = []
    # Zero matches: the row was not found. Two or more: ambiguous. Either way, nothing.
    if len(matches) != 1:
        return [], False
    label_words, dates = matches[0]
    return label_words + dates, any(find_date(w["text"]) == field.value for w in dates)


def as_date(field: FieldValue | None) -> date | None:
    """The field's value as a `date`, or None when it is empty or not a valid YYYY-MM-DD."""
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
        """Record the warning and mark every field involved as "Please verify"."""
        warnings.append(message)
        for field in fields:
            field.confidence = "review"

    # The licence's own three dates.
    dob, doi, doe = data.date_of_birth, data.date_of_issue, data.date_of_expiry
    birth, issue, expiry = as_date(dob), as_date(doi), as_date(doe)
    if birth and birth > today:
        flag(f"Date of birth ({birth}) is in the future.", dob)
    if issue and issue > today:
        flag(f"Date of issue ({issue}) is in the future.", doi)
    if issue and expiry and issue >= expiry:
        flag(f"Date of issue ({issue}) is not before the expiry date ({expiry}). Were they swapped?", doi, doe)
    if birth and issue and birth >= issue:
        flag(f"Date of birth ({birth}) is not before the date of issue ({issue}).", dob, doi)

    # Group the per-class fields by class: {"lmv": {"date_of_issue": ..., "valid_till": ...}}.
    per_class: dict[str, dict[str, FieldValue]] = {}
    for key, field in data.other_fields.items():
        if m := class_date_key(key):
            per_class.setdefault(m["cls"], {})[m["kind"]] = field
    # The same checks for each vehicle class.
    for cls, pair in per_class.items():
        label = cls.upper().replace("_", " ")
        cls_issue, cls_till = pair.get("date_of_issue"), pair.get("valid_till")
        start, end = as_date(cls_issue), as_date(cls_till)
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
# What the user saves from the form is checked before it is stored.

MAX_VALUE_CHARS = 1000
MAX_LABEL_CHARS = 64
# Names allowed for "other" fields: lowercase letters, digits and underscores, at most 64.
_OTHER_KEY = re.compile(r"[a-z0-9_]{1,64}")


def clean_user_data(data: LicenceData) -> LicenceData:
    """Validate and tidy user-edited data. Raises ValueError with a user-facing message."""
    data = data.model_copy(deep=True)  # never change the caller's object
    for key in data.other_fields:
        if not _OTHER_KEY.fullmatch(key):
            raise ValueError(f"Invalid field name '{key}'")
    for name, field in iter_fields(data):
        # Blank or whitespace-only input means "no value".
        value = (field.value or "").strip() or None
        if value and len(value) > MAX_VALUE_CHARS:
            raise ValueError(f"{name} is longer than {MAX_VALUE_CHARS} characters")
        # Dates may be typed in any supported format. They are stored as YYYY-MM-DD.
        if value and (name in DATE_FIELDS or class_date_key(name)):
            iso = normalize_date(value)
            if iso is None:
                raise ValueError(f"{name} must be a valid date, for example 15-06-2034")
            value = iso
        field.value = value
        # Only an item of other_fields that has a value keeps its printed label.
        label = " ".join((field.label or "").split()) or None
        if label and len(label) > MAX_LABEL_CHARS:
            raise ValueError(f"The label of {name} is longer than {MAX_LABEL_CHARS} characters")
        field.label = label if value and name.startswith("other_fields.") else None
    return data


# --- merge ----------------------------------------------------------------------------------
# The entry point: everything above is used here.


def iter_fields(data: LicenceData) -> list[tuple[str, FieldValue]]:
    """(name, field) for every field; other_fields are named "other_fields.<key>"."""
    core = [(name, getattr(data, name)) for name in CORE_FIELDS]
    return core + [(f"other_fields.{k}", v) for k, v in data.other_fields.items()]


def merge(data: LicenceData, ocr_text: str, words: list[dict] | None = None) -> tuple[LicenceData, list[str]]:
    """Cross-check every field against OCR: confidence from the text, bbox and score from the words.

    Per-class dates are anchored on their class's table row, and impossible date orders are
    flagged. Returns a new LicenceData plus warnings. Null-value fields are always "review".
    """
    data = data.model_copy(deep=True)  # never change the caller's object
    warnings: list[str] = []
    # Almost no OCR text means there is nothing to check against: every field stays "review".
    low_ocr = len(ocr_text.strip()) < MIN_OCR_CHARS
    if low_ocr:
        warnings.append(LOW_OCR_WARNING)

    words = words or []
    for name, field in iter_fields(data):
        per_class = class_date_key(name) is not None
        is_date = name in DATE_FIELDS or per_class
        # Start from "unconfirmed". The AI's own confidence and bbox are never trusted.
        field.confidence = "review"
        field.bbox = None
        field.confidence_score = None

        # Step 1: normalise dates to YYYY-MM-DD.
        if is_date and field.value is not None:
            # source_text may carry the printed label ("DOI: 16-06-2019"); a per-class row
            # holds two dates, so only the value itself is trusted there.
            iso = find_date(field.value) or (None if per_class else find_date(field.source_text))
            if iso:
                field.value = iso
            else:
                warnings.append(f"{name}: could not normalise the date '{field.value}'")

        # An empty field has nothing to confirm or to highlight.
        if field.value is None:
            continue
        # Step 2a: a per-class date is found on, and confirmed by, the table row of its class.
        if per_class:
            found, on_its_row = _class_row(field, words)
            field.bbox = _box_of(found) if found else None
            field.confidence_score = confidence_score(found)
            if on_its_row and not low_ocr:
                field.confidence = "high"
            continue
        # Step 2b: every other field is looked up by the text the AI copied from the card.
        anchor = field.source_text or field.value
        found = match_words(anchor, words) or match_words_parts(anchor, words)
        field.bbox = _box_of(found) if found else None
        field.confidence_score = confidence_score(found)
        if low_ocr:
            continue
        if text_matches_ocr(anchor, ocr_text, is_date=is_date) or parts_match_ocr(anchor, ocr_text):
            field.confidence = "high"

    # Step 3: dates that are printed but make no sense (swapped, in the future) go back to review.
    warnings += check_date_order(data)
    return data, warnings
