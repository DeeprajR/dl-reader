"""OCR with Tesseract.

`run_ocr` returns two things for one image:

  * the full printed text, used to confirm the AI model's answer and searched by the chat
  * every word with its position, used to draw the highlights on the document

Before Tesseract sees the image it is cleaned up: made grey, enlarged if small, and table
lines are erased (see `erase_rules` for why).
"""

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
# Never enlarge more than this: beyond it a tiny image only gets blurrier, not more readable.
_MAX_UPSCALE = 4.0


@dataclass
class OcrResult:
    """What OCR read from one image."""

    # All the text, one line of the card per line.
    text: str
    # {text, left, top, width, height, conf, line} in working-image pixels, reading order.
    words: list[dict]


def tesseract_cmd() -> str:
    """TESSERACT_CMD if it points at a real file, else rely on `tesseract` being on PATH."""
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and Path(cmd).is_file():
        return cmd
    # A path that does not exist is normal in Docker, where .env still holds a Windows path.
    if cmd:
        logger.warning("TESSERACT_CMD does not exist (%s); falling back to tesseract on PATH", cmd)
    return "tesseract"


def tesseract_version() -> str:
    """Raises if Tesseract cannot be run."""
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    return str(pytesseract.get_tesseract_version())


# --- erasing table lines ----------------------------------------------------------------------
# The helpers below work on a True/False image ("is this pixel dark?"). A "run" is a stretch
# of neighbouring True pixels in one row. Everything is done with whole-array NumPy operations,
# because a Python loop over millions of pixels would take seconds per image.


def _run_ids(mask: np.ndarray) -> np.ndarray:
    """Flat id (1..n) of the horizontal run each True pixel belongs to; 0 elsewhere.

    A False column is appended to every row so runs never continue onto the next row.
    """
    padded = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=bool)
    padded[:, :-1] = mask
    flat = padded.ravel()
    # A run starts at a True pixel whose left neighbour is False.
    starts = flat & ~np.concatenate(([False], flat[:-1]))
    # Counting the starts so far numbers the runs; multiplying by `flat` zeroes the background.
    return np.cumsum(starts) * flat


def _unpad(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Back from the flat, padded form used by `_run_ids` to an image of the original `shape`."""
    return flat.reshape(shape[0], shape[1] + 1)[:, :-1]


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """For each pixel of `mask`, the length of the horizontal run of True pixels it is in."""
    ids = _run_ids(mask)
    # bincount gives the size of each run; indexing it by `ids` hands every pixel its run's size.
    lengths = np.bincount(ids)[ids]
    lengths[ids == 0] = 0
    return _unpad(lengths, mask.shape)


def _thin_long_runs(mask: np.ndarray, thickness: np.ndarray, min_len: int, max_thickness: int) -> np.ndarray:
    """Horizontal runs at least `min_len` long that are thin (perpendicular `thickness` at most
    `max_thickness`) along at least 90% of their length: rulings, not letter gaps in banners."""
    ids = _run_ids(mask)
    padded = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=np.int32)
    padded[:, :-1] = thickness
    # Per run: how many pixels it has, and how many of those are thin.
    lengths = np.bincount(ids)
    thin = np.bincount(ids, weights=(padded.ravel() <= max_thickness) & (ids > 0))
    keep = (lengths >= min_len) & (thin >= 0.9 * lengths)
    keep[0] = False  # id 0 is the background, never a line
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
    # Sizes scale with the image: a line is at least 6% of the long side (120 px minimum) and
    # at most 0.8% of it thick (12 px minimum).
    long_side = max(pixels.shape)
    min_len, max_thickness = max(120, int(0.06 * long_side)), max(12, int(0.008 * long_side))
    # Run lengths in both directions. Transposing the image turns columns into rows, so the
    # same row-based helpers also measure vertical runs.
    across, down = _run_lengths(dark), _run_lengths(dark.T).T
    # Horizontal lines are long across and thin downwards; vertical lines the other way round.
    lines = _thin_long_runs(dark, down, min_len, max_thickness) | _thin_long_runs(
        dark.T, across.T, min_len, max_thickness
    ).T
    # Grow the lines by two pixels in every direction to also erase their soft, greyish edges.
    grown = lines.copy()
    for shift in ((1, 0), (-1, 0), (2, 0), (-2, 0), (0, 1), (0, -1), (0, 2), (0, -2)):
        grown |= np.roll(lines, shift, axis=(0, 1))
    pixels[grown] = 255
    return Image.fromarray(pixels)


# --- running Tesseract ------------------------------------------------------------------------


def _prepare(img: Image.Image) -> tuple[Image.Image, float]:
    """The image as Tesseract gets it, and the factor it was enlarged by (1.0 = not enlarged)."""
    # Transparent areas would turn black in greyscale: put the image on a white background.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(background, img)
    gray = ImageOps.grayscale(img)
    # Enlarge small images so the letters are big enough to read.
    long_side = max(gray.size)
    scale = 1.0
    if long_side < _TARGET_LONG_SIDE:
        scale = min(_MAX_UPSCALE, _TARGET_LONG_SIDE / long_side)
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.LANCZOS)
    return erase_rules(gray), scale


def run_ocr(image_path: Path | str) -> OcrResult:
    """Tesseract on the prepared image: the full text plus every word with its box and line.

    Boxes are scaled back to the working image's pixels, so they can be drawn on it directly.
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    with Image.open(image_path) as img:
        img.load()
        prepared, scale = _prepare(img)

    # image_to_data returns parallel lists: entry i of each list describes word i.
    data = pytesseract.image_to_data(prepared, output_type=pytesseract.Output.DICT)

    words: list[dict] = []
    # Tesseract numbers lines inside paragraphs inside blocks. Each (block, paragraph, line)
    # combination gets one running line number: 0, 1, 2, ...
    line_ids: dict[tuple[int, int, int], int] = {}
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        conf = float(data["conf"][i])
        # Entries without text, or with confidence -1, are layout boxes, not words.
        if not text or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line = line_ids.setdefault(key, len(line_ids))
        # Divide by `scale` to get back to the pixels of the image the user sees.
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

    # Rebuild the text from the words that were kept: one text line per OCR line.
    lines: dict[int, list[str]] = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w["text"])
    text = "\n".join(" ".join(parts) for parts in lines.values())
    return OcrResult(text=text, words=words)
