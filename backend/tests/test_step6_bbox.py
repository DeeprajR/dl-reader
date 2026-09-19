"""Step 6 (Phase 2) - bounding-box matching, field boxes and chat-source boxes.

Interface:
  extraction.match_bbox(target: str | None, words: list[dict]) -> Box | None   (pure function)
  extraction.merge(data, ocr_text, words=...) fills FieldValue.bbox and FieldValue.confidence_score
  chat sources carry the bbox of their chunk (fields and, best-effort, OCR text chunks)
The highlight overlay / click-to-highlight UI is verified in the browser.
"""

import pytest
from conftest import CANNED_OCR_TEXT, CANNED_WORDS, IMAGE_SIZE, fv, make_licence, words_from_lines

from app.schemas import Box

pytestmark = [pytest.mark.phase2, pytest.mark.step6]

# A small made-up card for the matching tests: four printed lines and their word positions.
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
    """An address printed over two lines gets one box that covers both lines."""
    from app.services.extraction import match_bbox

    box = match_bbox("Flat 302, Sai Residency,\nPune - 411052", WORDS)
    assert box == box_of("Flat", "Residency,", "Pune", "411052")
    assert box.h > 40  # spans two text lines


def test_bbox_ignores_punctuation_and_case():
    """Case and punctuation never matter, and a lone ":" between label and value does not break a match."""
    from app.services.extraction import match_bbox

    assert match_bbox("mh12 20190001234.", WORDS) == box_of("MH12", "20190001234")
    words = words_from_lines(["DOB : 12-08-1990"])
    assert match_bbox("12-08-1990", words) == box_of("12-08-1990", words=words)


def test_bbox_graceful_none():
    """Nothing to look for, or nothing to look in, gives None instead of an error."""
    from app.services.extraction import match_bbox

    assert match_bbox(None, WORDS) is None
    assert match_bbox("", WORDS) is None
    assert match_bbox("JOHN", []) is None
    assert match_bbox("JOHN", [{"text": ":", "left": 0, "top": 0, "width": 5, "height": 5, "conf": 90}]) is None


def test_merge_fills_field_boxes_from_ocr_words():
    """merge gives each field its box, and none to a value that is not on the card or is empty."""
    from app.services.extraction import merge

    data = make_licence(issuing_authority=fv("NOT ON THE CARD"), vehicle_classes=fv(None))
    merged, _ = merge(data, CANNED_OCR_TEXT, words=CANNED_WORDS)
    assert merged.full_name.bbox == box_of("JOHN", "DOE", words=CANNED_WORDS)
    assert merged.licence_number.bbox is not None
    assert merged.issuing_authority.bbox is None  # no OCR match -> snippet only
    assert merged.vehicle_classes.bbox is None  # null value


def test_extract_endpoint_returns_boxes_inside_the_image(client, extracted):
    """Through the API: nearly every field gets a box, and every box lies inside the image."""
    data = client.get(f"/api/documents/{extracted}/extract").json()["data"]
    boxes = [f["bbox"] for f in [*(v for k, v in data.items() if k != "other_fields"), *data["other_fields"].values()]]
    found = [b for b in boxes if b]
    assert len(found) >= len(boxes) - 1  # most printed fields get a highlight
    width, height = IMAGE_SIZE
    for b in found:
        assert 0 <= b["x"] and 0 <= b["y"] and b["x"] + b["w"] <= width and b["y"] + b["h"] <= height


def test_chat_sources_carry_boxes(client, extracted, monkeypatch):
    """Chat sources can be highlighted too: field sources always, document-text sources when they are found."""
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
    """A column value ("LMV, MCWG") is never one run of words, so each part is located on its own."""
    from app.services.extraction import match_bbox, match_bbox_parts

    assert match_bbox("LMV\nMCWG", TABLE_WORDS) is None  # never consecutive, LMV unread
    assert match_bbox_parts("LMV\nMCWG", TABLE_WORDS) == box_of("/MCWG", words=TABLE_WORDS)
    assert match_bbox_parts("LMV, MCWG", TABLE_WORDS) == box_of("/MCWG", words=TABLE_WORDS)


def test_part_fallback_guards():
    """The part-by-part fallback stays out of the way: single values, very short parts, nothing found, None."""
    from app.services.extraction import match_bbox_parts

    assert match_bbox_parts("MCWG", TABLE_WORDS) is None  # a single part is not a fallback case
    assert match_bbox_parts("A+\nO+", TABLE_WORDS) is None  # parts under 3 chars are ignored
    assert match_bbox_parts("XYZ\nQRS", TABLE_WORDS) is None  # nothing found
    assert match_bbox_parts(None, TABLE_WORDS) is None


def test_merge_uses_part_fallback_only_when_whole_value_fails():
    """The fallback is used only after the normal match fails, and a half-found value stays under review."""
    from app.services.extraction import merge

    data = make_licence(vehicle_classes=fv("LMV, MCWG", "LMV\nMCWG"))

    # Whole-value matches are untouched by the fallback.
    merged, _ = merge(data, CANNED_OCR_TEXT, words=CANNED_WORDS)
    assert merged.full_name.bbox == box_of("JOHN", "DOE", words=CANNED_WORDS)

    # Only the table is printed: the column value falls back to the cell OCR could read.
    merged, _ = merge(data, "\n".join(TABLE_LINES), words=TABLE_WORDS)
    assert merged.vehicle_classes.bbox == box_of("/MCWG", words=TABLE_WORDS)
    assert merged.vehicle_classes.confidence == "review"  # LMV is still unconfirmed by OCR


def row_box(line, words=TABLE_WORDS):
    """Union box of the non-punctuation words on one synthetic line."""
    row = [w for w in words if w["line"] == line and w["text"] not in {"|"}]
    left, top = min(w["left"] for w in row), min(w["top"] for w in row)
    right, bottom = max(w["left"] + w["width"] for w in row), max(w["top"] + w["height"] for w in row)
    return Box(x=left, y=top, w=right - left, h=bottom - top)


def test_per_class_dates_are_anchored_on_their_own_row():
    """A class's date is looked up on that class's table row, and confirmed only if it is printed there."""
    from app.services.extraction import merge

    other = {
        "mcwg_valid_till": fv("2034-06-15", "MCWG 16-06-2019 15-06-2034"),
        "mcwg_date_of_issue": fv("2030-01-01", "MCWG 16-06-2019 15-06-2034"),  # not printed on its row
        "lmv_valid_till": fv("2034-06-15", "LMV 16-06-2019 15-06-2034"),  # LMV cell unreadable
    }
    merged, _ = merge(make_licence(other_fields=other), "\n".join(TABLE_LINES), words=TABLE_WORDS)
    mcwg, wrong, lmv = (merged.other_fields[k] for k in other)
    assert mcwg.bbox == row_box(2) and mcwg.confidence == "high"  # MCWG row, date on that row
    assert wrong.bbox == row_box(2) and wrong.confidence == "review"
    # The LMV row text would fuzzy-match the MCWG row (same dates, shared "M"): no box beats a wrong one.
    assert lmv.bbox is None and lmv.confidence == "review"


def test_labelled_date_source_highlights_label_and_date():
    """A source such as "DOI: 16-06-2019" highlights the label together with the date."""
    from app.services.extraction import match_bbox

    assert match_bbox("DOI: 16-06-2019", CANNED_WORDS) == box_of("DOI", "16-06-2019", words=CANNED_WORDS)


# Many classes, some with similar codes, each with its own dates.
MANY_ROWS = {
    "mcwog": "MCWOG 01-02-2010 31-01-2030",
    "mcwg": "MCWG 05-06-2015 04-06-2035",
    "lmv": "LMV 10-11-2016 09-11-2036",
    "lmv_nt": "LMV NT 10-11-2016 09-11-2036",
    "lmv_tr": "LMV TR 12-03-2018 11-03-2021",
}


def many_class_merge(rows):
    """Build a class table from `rows`, run merge on it, and return the per-class fields and the words."""
    from app.services.extraction import merge

    lines = ["Class of Vehicle DOI Valid Till", *rows.values()]
    words = words_from_lines(lines, x0=20, y0=100, line_h=30, char_w=10, h=20)
    other = {}
    for cls, row in rows.items():
        issue, till = row.split()[-2:]
        other[f"{cls}_date_of_issue"] = fv(issue, MANY_ROWS[cls])
        other[f"{cls}_valid_till"] = fv(till, MANY_ROWS[cls])
    merged, _ = merge(make_licence(other_fields=other), "\n".join(lines), words=words)
    return merged.other_fields, words


def test_similar_class_codes_each_anchor_on_their_own_row():
    """LMV, LMV NT, LMV TR, MCWG and MCWOG look alike, yet each finds exactly its own row."""
    fields, words = many_class_merge(MANY_ROWS)
    for i, cls in enumerate(MANY_ROWS, start=1):  # line 0 is the header
        for kind in ("date_of_issue", "valid_till"):
            field = fields[f"{cls}_{kind}"]
            assert field.bbox == row_box(i, words), f"{cls}_{kind}"
            assert field.confidence == "high"


def test_garbled_class_code_gets_no_box_instead_of_a_similar_row():
    """When OCR garbles a class code, the field gets no box rather than the box of a similar-looking class."""
    rows = {**MANY_ROWS, "mcwg": "MCW6 05-06-2015 04-06-2035"}  # OCR misread; MCWOG is 0.89 similar
    fields, _ = many_class_merge(rows)
    assert fields["mcwg_valid_till"].bbox is None
    assert fields["mcwg_valid_till"].confidence == "review"
    assert fields["mcwog_valid_till"].confidence == "high"


# --- confidence score: how sure OCR was about the words a value was found in ------------------


def with_conf(words, **conf_by_text):
    """A copy of `words` in which the named words get the given OCR confidence."""
    return [{**w, "conf": conf_by_text.get(w["text"], w["conf"])} for w in words]


def test_confidence_score_is_the_average_ocr_confidence_of_the_matched_words():
    """The score is Tesseract's average certainty (0-100) about exactly the words the value matched."""
    from app.services.extraction import confidence_score, merge

    assert confidence_score([{"conf": 90.0}, {"conf": 72.0}]) == 81
    assert confidence_score([{"conf": 96.4}]) == 96
    assert confidence_score([]) is None

    words = with_conf(CANNED_WORDS, JOHN=60.0, DOE=80.0)
    merged, _ = merge(make_licence(), CANNED_OCR_TEXT, words=words)
    assert merged.full_name.confidence_score == 70  # only JOHN and DOE count
    assert merged.licence_number.confidence_score == 95  # every other word is read at 95


def test_confidence_score_is_none_when_the_value_is_not_located():
    """No matched words, no score: a value that is not on the card, an empty field, or no OCR words."""
    from app.services.extraction import merge

    data = make_licence(issuing_authority=fv("NOT ON THE CARD"), vehicle_classes=fv(None))
    merged, _ = merge(data, CANNED_OCR_TEXT, words=CANNED_WORDS)
    assert merged.issuing_authority.confidence_score is None
    assert merged.vehicle_classes.confidence_score is None

    merged, _ = merge(make_licence(), CANNED_OCR_TEXT)  # OCR text only, no word list
    assert merged.full_name.confidence == "high" and merged.full_name.confidence_score is None


def test_confidence_score_never_trusts_the_models_own_number():
    """A score sent by the AI model is discarded: the score only ever comes from OCR."""
    from app.schemas import FieldValue
    from app.services.extraction import merge

    boasting = FieldValue(value="NOT ON THE CARD", source_text="NOT ON THE CARD", confidence="high",
                          bbox=None, confidence_score=100)
    merged, _ = merge(make_licence(full_name=boasting), CANNED_OCR_TEXT, words=CANNED_WORDS)
    assert merged.full_name.confidence_score is None and merged.full_name.confidence == "review"


def test_per_class_dates_get_the_score_of_their_own_row():
    """A per-class date is scored from its class's table row, not from a row with the same dates."""
    fields, _ = many_class_merge(MANY_ROWS)
    assert fields["lmv_tr_valid_till"].confidence_score == 95
    garbled, _ = many_class_merge({**MANY_ROWS, "mcwg": "MCW6 05-06-2015 04-06-2035"})
    assert garbled["mcwg_valid_till"].confidence_score is None  # row not found: no score


def test_extract_endpoint_returns_confidence_scores(client, extracted):
    """Through the API: located fields carry a 0-100 score, and the score survives a save."""
    data = client.get(f"/api/documents/{extracted}/extract").json()["data"]
    assert data["full_name"]["confidence_score"] == 95
    assert "page" not in data["full_name"]  # the unused page number is gone

    data["full_name"]["value"] = "JOHN A DOE"
    saved = client.put(f"/api/documents/{extracted}/data", json=data).json()
    assert saved["full_name"]["confidence_score"] == 95


def test_rows_are_found_by_position_even_when_ocr_splits_the_cells():
    """Rows are rebuilt from where the words are, not from Tesseract's line numbers, which break in tables."""
    from app.services.extraction import merge

    rows = {"mcwg": "MCWG 05-06-2015 04-06-2035", "lmv": "LMV 10-11-2016 09-11-2036"}
    words = words_from_lines(["Class DOI Valid Till", *rows.values()], x0=20, y0=100, line_h=30, char_w=10, h=20)
    for i, w in enumerate(words):
        w["line"] = 100 + i  # every cell its own Tesseract line, as after erasing the rulings
    other = {f"{c}_valid_till": fv(r.split()[-1], r) for c, r in rows.items()}
    merged, _ = merge(make_licence(other_fields=other), "\n".join(rows.values()), words=words)
    assert merged.other_fields["mcwg_valid_till"].bbox == row_box(1, words_from_lines(["Class DOI Valid Till", *rows.values()], x0=20, y0=100, line_h=30, char_w=10, h=20))
    assert merged.other_fields["lmv_valid_till"].confidence == "high"


def test_side_by_side_cards_do_not_mix_rows():
    """Front and back photographed side by side: text at the same height on the other card is not mixed in."""
    from app.services.extraction import merge

    front = words_from_lines(["DOB : 03-09-1992"], x0=20, y0=250)
    back = words_from_lines(["MCWG 18-01-2017 17-01-2037"], x0=900, y0=246)  # same height, other card
    other = {"mcwg_date_of_issue": fv("2017-01-18", "MCWG 18-01-2017 17-01-2037")}
    merged, _ = merge(make_licence(other_fields=other), "DOB : 03-09-1992 MCWG 18-01-2017 17-01-2037", words=front + back)
    field = merged.other_fields["mcwg_date_of_issue"]
    assert field.confidence == "high"
    assert field.bbox.x >= 900  # label and dates on the back card only


def test_multi_part_values_are_confirmed_part_by_part():
    """A multi-part value is confirmed only when every part is printed, as a whole word."""
    from app.services.extraction import parts_match_ocr

    table = "Class of Vehicle DOI Valid Till\nLMV 16-06-2019 15-06-2034\nMCWG 16-06-2019 15-06-2034"
    assert parts_match_ocr("LMV\nMCWG", table)
    assert parts_match_ocr("LMV, MCWG", table)
    assert not parts_match_ocr("LMV, HGMV", table)  # a class that is not printed
    assert not parts_match_ocr("A, B", "A B C")  # too short to check safely
    assert not parts_match_ocr("LMV", table)  # single part: the normal check applies
