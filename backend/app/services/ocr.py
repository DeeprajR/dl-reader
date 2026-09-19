"""Tesseract wrapper: returns the full text AND word-level boxes (via image_to_data)."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


def _run_ids(mask: np.ndarray) -> np.ndarray:
    """Flat id (1..n) of the horizontal run each True pixel belongs to; 0 elsewhere.

    A False column is appended to every row so runs never continue onto the next row.
    """
    padded = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=bool)
    padded[:, :-1] = mask
    flat = padded.ravel()
    starts = flat & ~np.concatenate(([False], flat[:-1]))
    return np.cumsum(starts) * flat


def _unpad(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return flat.reshape(shape[0], shape[1] + 1)[:, :-1]


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """For each pixel of `mask`, the length of the horizontal run of True pixels it is in."""
    ids = _run_ids(mask)
    lengths = np.bincount(ids)[ids]
    lengths[ids == 0] = 0
    return _unpad(lengths, mask.shape)


def _thin_long_runs(mask: np.ndarray, thickness: np.ndarray, min_len: int, max_thickness: int) -> np.ndarray:
    """Horizontal runs at least `min_len` long that are thin (perpendicular `thickness` at most
    `max_thickness`) along at least 90% of their length: rulings, not letter gaps in banners."""
    ids = _run_ids(mask)
    padded = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=np.int32)
    padded[:, :-1] = thickness
    lengths = np.bincount(ids)
    thin = np.bincount(ids, weights=(padded.ravel() <= max_thickness) & (ids > 0))
    keep = (lengths >= min_len) & (thin >= 0.9 * lengths)
    keep[0] = False
    return _unpad(keep[ids], mask.shape)


def erase_rules(gray: Image.Image) -> Image.Image:
    """Whiten long, thin straight lines (table rulings, card borders) before OCR.

    Tesseract's layout analysis treats ruled table regions as graphics and drops whole rows
    (on the samples: the LMV row, and half of a six-class table). A ruling is far longer than
    any letter stroke and thin along (nearly) its whole length; solid banners with white title
    text are thick, and the dark gaps between their letters are thin only beside the letters,
    so both are kept. Geometry is unchanged, so word boxes still match the image.
    """
    pixels = np.asarray(gray).copy()
    dark = pixels < 140
    long_side = max(pixels.shape)
    min_len, max_thickness = max(120, int(0.06 * long_side)), max(12, int(0.008 * long_side))
    across, down = _run_lengths(dark), _run_lengths(dark.T).T
    lines = _thin_long_runs(dark, down, min_len, max_thickness) | _thin_long_runs(
        dark.T, across.T, min_len, max_thickness
    ).T
    grown = lines.copy()
    for shift in ((1, 0), (-1, 0), (2, 0), (-2, 0), (0, 1), (0, -1), (0, 2), (0, -2)):
        grown |= np.roll(lines, shift, axis=(0, 1))
    pixels[grown] = 255
    return Image.fromarray(pixels)


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
    return erase_rules(gray), scale


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
