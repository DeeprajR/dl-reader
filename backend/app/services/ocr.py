"""Tesseract wrapper: returns the full text AND word-level boxes (via image_to_data)."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Tesseract reads best when glyphs are ~30px tall; small scans and screenshots are
# upscaled before OCR and the boxes are mapped back to working-image pixels.
_TARGET_LONG_SIDE = 2000
_MAX_UPSCALE = 4.0


@dataclass
class OcrResult:
    text: str
    # {text, left, top, width, height, conf, line} in working-image pixels, reading order.
    words: list[dict]


def tesseract_cmd() -> str:
    """TESSERACT_CMD if it points at a real file, else rely on `tesseract` being on PATH."""
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and Path(cmd).is_file():
        return cmd
    if cmd:
        logger.warning("TESSERACT_CMD does not exist (%s); falling back to tesseract on PATH", cmd)
    return "tesseract"


def tesseract_version() -> str:
    """Raises if Tesseract cannot be run."""
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    return str(pytesseract.get_tesseract_version())


def _prepare(img: Image.Image) -> tuple[Image.Image, float]:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(background, img)
    gray = ImageOps.grayscale(img)
    long_side = max(gray.size)
    scale = 1.0
    if long_side < _TARGET_LONG_SIDE:
        scale = min(_MAX_UPSCALE, _TARGET_LONG_SIDE / long_side)
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.LANCZOS)
    return gray, scale


def run_ocr(image_path: Path | str) -> OcrResult:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    with Image.open(image_path) as img:
        img.load()
        prepared, scale = _prepare(img)

    data = pytesseract.image_to_data(prepared, output_type=pytesseract.Output.DICT)

    words: list[dict] = []
    line_ids: dict[tuple[int, int, int], int] = {}
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line = line_ids.setdefault(key, len(line_ids))
        words.append(
            {
                "text": text,
                "left": round(data["left"][i] / scale),
                "top": round(data["top"][i] / scale),
                "width": max(1, round(data["width"][i] / scale)),
                "height": max(1, round(data["height"][i] / scale)),
                "conf": conf,
                "line": line,
            }
        )

    lines: dict[int, list[str]] = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w["text"])
    text = "\n".join(" ".join(parts) for parts in lines.values())
    return OcrResult(text=text, words=words)
