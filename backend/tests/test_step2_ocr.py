"""Step 2 - OCR service against the real Tesseract binary (skipped when it is not installed)."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services import ocr

pytestmark = [pytest.mark.phase1, pytest.mark.step2]


@pytest.fixture(scope="module", autouse=True)
def require_tesseract():
    try:
        ocr.tesseract_version()
    except Exception:
        pytest.skip("Tesseract not installed (set TESSERACT_CMD or add it to PATH)")


def render(tmp_path, text, size=(700, 120), origin=(20, 30), font_size=36, mode="RGB", background="white"):
    img = Image.new(mode, size, background)
    font = ImageFont.load_default(size=font_size)
    ImageDraw.Draw(img).text(origin, text, fill="black", font=font)
    path = tmp_path / "doc.png"
    img.save(path)
    return path, font


def test_ocr_returns_text_and_word_boxes_in_working_image_pixels(tmp_path):
    prefix, number = "DL No: MH12 ", "20190001234"
    path, font = render(tmp_path, prefix + number)

    result = ocr.run_ocr(path)
    assert "MH12" in result.text and number in result.text
    assert set(result.words[0]) >= {"text", "left", "top", "width", "height", "conf"}

    # Small images are upscaled for Tesseract; boxes must come back in original pixels.
    word = next(w for w in result.words if number in w["text"])
    expected_left = 20 + ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(prefix, font=font)
    assert abs(word["left"] - expected_left) < 12
    assert 20 <= word["top"] <= 60
    assert word["left"] + word["width"] <= 700 and word["top"] + word["height"] <= 120


def test_ocr_text_keeps_line_structure(tmp_path):
    path, _ = render(tmp_path, "Name : JOHN DOE\nDOB : 12-08-1990", size=(600, 200))
    lines = ocr.run_ocr(path).text.splitlines()
    assert any("JOHN DOE" in line for line in lines)
    assert any("12-08-1990" in line for line in lines)
    assert not any("JOHN" in line and "1990" in line for line in lines)


def test_ocr_handles_transparent_png(tmp_path):
    # Transparent *black* background: naive grayscale would give black text on black.
    path, _ = render(tmp_path, "LICENCE 12345", mode="RGBA", background=(0, 0, 0, 0))
    assert "12345" in ocr.run_ocr(path).text


def ruled_table(tmp_path, rows):
    """A bordered table like the licences' class tables (Tesseract drops rows of these)."""
    font = ImageFont.load_default(size=24)
    img = Image.new("RGB", (760, 60 + 44 * len(rows)), "white")
    d = ImageDraw.Draw(img)
    for r, row in enumerate(rows):
        x = 20
        for cell, width in zip(row, (280, 220, 220)):
            d.rectangle([x, 20 + r * 44, x + width, 20 + (r + 1) * 44], outline="black", width=2)
            d.text((x + 12, 30 + r * 44), cell, fill="black", font=font)
            x += width
    path = tmp_path / "table.png"
    img.save(path)
    return path


def test_ocr_reads_every_row_of_a_ruled_table(tmp_path):
    rows = [("Class of Vehicle", "DOI", "Valid Till"), ("MCWOG", "01-02-2010", "31-01-2030"),
            ("MCWG", "05-06-2015", "31-01-2030"), ("LMV", "10-11-2016", "31-01-2030"), ("LMV-TR", "12-03-2018", "11-03-2021")]
    text = ocr.run_ocr(ruled_table(tmp_path, rows)).text
    for cls, issued, _ in rows[1:]:
        assert cls in text and issued in text, cls


def test_erase_rules_removes_lines_but_keeps_text():
    import numpy as np

    img = Image.new("L", (1000, 300), 255)
    d = ImageDraw.Draw(img)
    d.line([(10, 150), (990, 150)], fill=0, width=3)  # table ruling
    d.line([(500, 10), (500, 290)], fill=0, width=3)
    d.text((40, 40), "LMV 16-06-2019", fill=0, font=ImageFont.load_default(size=40))
    out = np.asarray(ocr.erase_rules(img))
    assert (out[148:153, :] < 140).sum() == 0  # horizontal ruling gone
    assert (out[:, 498:503] < 140).sum() == 0  # vertical ruling gone
    text_region = (slice(30, 100), slice(30, 450))
    assert (out[text_region] < 140).sum() == (np.asarray(img)[text_region] < 140).sum()  # text untouched


def test_erase_rules_keeps_white_text_on_a_dark_banner(tmp_path):
    import numpy as np

    img = Image.new("RGB", (900, 160), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 880, 110], fill=(200, 30, 40))  # red title bar, like the Maharashtra card
    d.text((60, 45), "MAHARASHTRA STATE MOTOR LICENCE", fill="white", font=ImageFont.load_default(size=36))
    gray = img.convert("L")
    out = np.asarray(ocr.erase_rules(gray))
    assert (out[20:110, 20:880] < 140).sum() >= 0.98 * (np.asarray(gray)[20:110, 20:880] < 140).sum()

    path = tmp_path / "banner.png"
    img.save(path)
    assert "MAHARASHTRA STATE" in ocr.run_ocr(path).text
