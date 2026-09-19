"""Step 6 (Phase 2) - bounding-box matching, field boxes and chat-source boxes.

Acceptance tests written ahead of the code; pending until 6 is in PHASE2_STEPS_DONE.
Interface defined here:
  extraction.match_bbox(target: str | None, words: list[dict]) -> Box | None   (pure function)
  extraction.merge(data, ocr_text, words=..., page=1) fills FieldValue.bbox
  chat sources carry the bbox of their chunk (fields and, best-effort, OCR text chunks)
The highlight overlay / click-to-highlight UI is verified in the browser.
"""

import pytest
from conftest import CANNED_OCR_TEXT, CANNED_WORDS, IMAGE_SIZE, fv, make_licence, words_from_lines

from app.schemas import Box

pytestmark = [pytest.mark.phase2, pytest.mark.step6]

LINES = ["Name : JOHN DOE", "DL No: MH12 20190001234", "Address : Flat 302, Sai Residency,", "Pune - 411052"]
WORDS = words_from_lines(LINES, x0=10, y0=10, line_h=40, char_w=10, h=20)


def box_of(*texts, words=WORDS):
    """Union box of the listed word texts (first occurrence of each)."""
    picked = [next(w for w in words if w["text"] == t) for t in texts]
    left, top = min(w["left"] for w in picked), min(w["top"] for w in picked)
    right = max(w["left"] + w["width"] for w in picked)
    bottom = max(w["top"] + w["height"] for w in picked)
    return Box(x=left, y=top, w=right - left, h=bottom - top)


def test_bbox_matching():
    """Spec test 3: exact match; fuzzy >= 0.8; no match -> None; multi-word target unions boxes."""
    from app.services.extraction import match_bbox

    assert match_bbox("JOHN", WORDS) == box_of("JOHN")  # exact
    assert match_bbox("JOHN DOF", WORDS) == box_of("JOHN", "DOE")  # fuzzy (one char off)
    assert match_bbox("COMPLETELY ABSENT TEXT", WORDS) is None  # no match
    assert match_bbox("MH12 20190001234", WORDS) == box_of("MH12", "20190001234")  # union


def test_bbox_multiline_address_covers_all_lines():
    from app.services.extraction import match_bbox

    box = match_bbox("Flat 302, Sai Residency,\nPune - 411052", WORDS)
    assert box == box_of("Flat", "Residency,", "Pune", "411052")
    assert box.h > 40  # spans two text lines


def test_bbox_ignores_punctuation_and_case():
    from app.services.extraction import match_bbox

    assert match_bbox("mh12 20190001234.", WORDS) == box_of("MH12", "20190001234")
    words = words_from_lines(["DOB : 12-08-1990"])
    assert match_bbox("12-08-1990", words) == box_of("12-08-1990", words=words)


def test_bbox_graceful_none():
    from app.services.extraction import match_bbox

    assert match_bbox(None, WORDS) is None
    assert match_bbox("", WORDS) is None
    assert match_bbox("JOHN", []) is None
    assert match_bbox("JOHN", [{"text": ":", "left": 0, "top": 0, "width": 5, "height": 5, "conf": 90}]) is None


def test_merge_fills_field_boxes_from_ocr_words():
    from app.services.extraction import merge

    data = make_licence(issuing_authority=fv("NOT ON THE CARD"), vehicle_classes=fv(None))
    merged, _ = merge(data, CANNED_OCR_TEXT, words=CANNED_WORDS)
    assert merged.full_name.bbox == box_of("JOHN", "DOE", words=CANNED_WORDS)
    assert merged.licence_number.bbox is not None
    assert merged.issuing_authority.bbox is None  # no OCR match -> snippet only
    assert merged.vehicle_classes.bbox is None  # null value


def test_extract_endpoint_returns_boxes_inside_the_image(client, extracted):
    data = client.get(f"/api/documents/{extracted}/extract").json()["data"]
    boxes = [f["bbox"] for f in [*(v for k, v in data.items() if k != "other_fields"), *data["other_fields"].values()]]
    found = [b for b in boxes if b]
    assert len(found) >= len(boxes) - 1  # most printed fields get a highlight
    width, height = IMAGE_SIZE
    for b in found:
        assert 0 <= b["x"] and 0 <= b["y"] and b["x"] + b["w"] <= width and b["y"] + b["h"] <= height


def test_chat_sources_carry_boxes(client, extracted, monkeypatch):
    from app.services import rag

    async def reply(client_, model, messages):
        return "The licence number is MH12 20190001234."

    monkeypatch.setattr(rag, "complete", reply)
    sources = client.post(f"/api/documents/{extracted}/chat", json={"question": "What is the licence number?"}).json()[
        "sources"
    ]
    fields = [s for s in sources if s["origin"] == "extracted_fields"]
    texts = [s for s in sources if s["origin"] == "ocr_text"]
    assert fields and all(s["bbox"] for s in fields)
    assert texts and any(s["bbox"] for s in texts)  # OCR chunks matched best-effort


# Table columns: the value's parts are not consecutive in OCR reading order (rows are read
# left to right), and Tesseract may miss some cells entirely - as on the Maharashtra sample.
TABLE_LINES = ["Valid for Class of Vehicle | DOI | Valid Till", "oe 16-06-2019 | 15-06-2034", "/MCWG 16-06-2019 | 15-06-2034"]
TABLE_WORDS = words_from_lines(TABLE_LINES, x0=20, y0=300, line_h=30, char_w=10, h=20)


def test_table_column_highlights_the_cells_ocr_could_read():
    from app.services.extraction import match_bbox, match_bbox_parts

    assert match_bbox("LMV\nMCWG", TABLE_WORDS) is None  # never consecutive, LMV unread
    assert match_bbox_parts("LMV\nMCWG", TABLE_WORDS) == box_of("/MCWG", words=TABLE_WORDS)
    assert match_bbox_parts("LMV, MCWG", TABLE_WORDS) == box_of("/MCWG", words=TABLE_WORDS)


def test_part_fallback_guards():
    from app.services.extraction import match_bbox_parts

    assert match_bbox_parts("MCWG", TABLE_WORDS) is None  # a single part is not a fallback case
    assert match_bbox_parts("A+\nO+", TABLE_WORDS) is None  # parts under 3 chars are ignored
    assert match_bbox_parts("XYZ\nQRS", TABLE_WORDS) is None  # nothing found
    assert match_bbox_parts(None, TABLE_WORDS) is None


def test_merge_uses_part_fallback_only_when_whole_value_fails():
    from app.services.extraction import merge

    data = make_licence(vehicle_classes=fv("LMV, MCWG", "LMV\nMCWG"))

    # Whole-value matches are untouched by the fallback.
    merged, _ = merge(data, CANNED_OCR_TEXT, words=CANNED_WORDS)
    assert merged.full_name.bbox == box_of("JOHN", "DOE", words=CANNED_WORDS)

    # Only the table is printed: the column value falls back to the cell OCR could read.
    merged, _ = merge(data, "\n".join(TABLE_LINES), words=TABLE_WORDS)
    assert merged.vehicle_classes.bbox == box_of("/MCWG", words=TABLE_WORDS)
    assert merged.vehicle_classes.confidence == "review"  # LMV is still unconfirmed by OCR
