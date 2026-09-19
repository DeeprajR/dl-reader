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
