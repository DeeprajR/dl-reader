"""Compare two vision models on the sample licences.

Usage (from backend/):  python scripts/compare_models.py

Runs every jpg/jpeg/png/pdf in ./samples (repo root) through LLM_MODEL and LLM_MODEL_ALT using
the same provider code path as the app, then prints a field-by-field diff table per document
plus per-model null-field counts and total latency. There is no pass/fail logic: the output is
for a human to judge which model should be the default.
"""

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path

# The script lives outside the `app` package, so the backend folder is added to the import path.
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def _reexec_in_venv() -> None:
    """Run with backend/.venv's Python when started from another interpreter.

    Saves the usual "ModuleNotFoundError: pdf2image" when the venv is not activated.
    """
    if sys.prefix != sys.base_prefix:
        return  # already inside a virtual environment
    venv_dir = BACKEND_DIR / ".venv"
    candidates = (venv_dir / "Scripts" / "python.exe", venv_dir / "bin" / "python")
    python = next((p for p in candidates if p.is_file()), None)
    if python is None:
        return
    print(f"Using {python}", file=sys.stderr)
    if os.name == "nt":  # os.execv on Windows detaches from the console; wait instead
        import subprocess

        sys.exit(subprocess.call([str(python), *sys.argv]))
    os.execv(str(python), [str(python), *sys.argv])


_reexec_in_venv()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(REPO_DIR / ".env")

from app.routes.documents import _EXTENSIONS, image_for_llm, prepare_working_image  # noqa: E402
from app.schemas import LicenceData  # noqa: E402
from app.services.extraction import iter_fields, normalize  # noqa: E402
from app.services.providers.base import DEFAULT_MODEL, get_provider  # noqa: E402

DEFAULT_ALT_MODEL = "anthropic/claude-sonnet-5"
SAMPLES_DIR = REPO_DIR / "samples"
# Column widths of the printed table, in characters.
COL_FIELD, COL_VALUE, COL_AGREE = 30, 36, 6


def load_sample(path: Path) -> tuple[bytes, str]:
    """Same preparation as an upload + extraction in the app (PDF pages 1-2, LLM downscale)."""
    data = path.read_bytes()
    image, _, media_type, _, _ = prepare_working_image(_EXTENSIONS[path.suffix.lower()], data)
    return image_for_llm(image if image is not None else data, media_type)


async def run_model(model: str, image: bytes, media_type: str):
    """Read one sample with one model. Returns (data, error message, seconds); a failure is reported, not raised."""
    start = time.perf_counter()
    try:
        data = await get_provider(model).extract(image, media_type)
        return data, None, time.perf_counter() - start
    except Exception as e:  # report and keep going; no pass/fail
        return None, f"{type(e).__name__}: {e}", time.perf_counter() - start


def values(data: LicenceData | None) -> dict[str, str | None]:
    """Every field's value by name, e.g. {"full_name": "JOHN DOE", ...}. Empty when the model failed."""
    return {name: field.value for name, field in iter_fields(data)} if data else {}


def print_row(cells: list[str], widths: list[int]) -> None:
    """Print one table row. A cell that is too long for its column wraps onto further lines."""
    wrapped = [textwrap.wrap(c, w) or [""] for c, w in zip(cells, widths)]
    for i in range(max(len(w) for w in wrapped)):
        print(" | ".join((w[i] if i < len(w) else "").ljust(width) for w, width in zip(wrapped, widths)))


def print_table(label_a: str, label_b: str, a: dict, b: dict) -> None:
    """Print both models' values field by field, with a yes/NO column showing where they agree."""
    widths = [COL_FIELD, COL_VALUE, COL_VALUE, COL_AGREE]
    # All field names, in model A's order, followed by the fields that only model B returned.
    names = list(a) + [n for n in b if n not in a]
    print_row(["field", label_a, label_b, "agree?"], widths)
    print("-+-".join("-" * w for w in widths))
    for name in names:
        va, vb = a.get(name), b.get(name)
        # Compared after normalising, so "John Doe" and "JOHN DOE." count as agreeing.
        agree = "yes" if normalize(va) == normalize(vb) else "NO"
        print_row([name, "(null)" if va is None else va, "(null)" if vb is None else vb, agree], widths)


async def main() -> None:
    """Read every sample with both models, then print a table per document and a summary."""
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8")
    model_a = os.getenv("LLM_MODEL") or DEFAULT_MODEL
    model_b = os.getenv("LLM_MODEL_ALT") or DEFAULT_ALT_MODEL
    samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in _EXTENSIONS)
    if not samples:
        sys.exit(f"No .jpg/.jpeg/.png/.pdf files found in {SAMPLES_DIR}")

    print(f"Model A: {model_a}\nModel B: {model_b}\nSamples: {len(samples)} in {SAMPLES_DIR}\n")
    # Running totals per model, for the summary at the end.
    totals = {m: {"latency": 0.0, "null_core": 0, "null_all": 0, "fields": 0, "errors": 0} for m in (model_a, model_b)}

    for path in samples:
        image, media_type = load_sample(path)
        # Both models read the same image at the same time.
        (data_a, err_a, t_a), (data_b, err_b, t_b) = await asyncio.gather(
            run_model(model_a, image, media_type), run_model(model_b, image, media_type)
        )
        print(f"=== {path.name}   (A: {t_a:.1f}s, B: {t_b:.1f}s)")
        for model, err in ((model_a, err_a), (model_b, err_b)):
            if err:
                print(f"  !! {model} failed: {err}")
        print_table("A: " + model_a, "B: " + model_b, values(data_a), values(data_b))
        print()

        for model, data, err, t in ((model_a, data_a, err_a, t_a), (model_b, data_b, err_b, t_b)):
            tot = totals[model]
            tot["latency"] += t
            if err:
                tot["errors"] += 1
                continue
            # Count the empty fields: all of them, and the eight main ones (names without the "other_fields." prefix).
            vals = values(data)
            tot["fields"] += len(vals)
            tot["null_all"] += sum(v is None for v in vals.values())
            tot["null_core"] += sum(v is None for n, v in vals.items() if not n.startswith("other_fields."))

    print("=== Summary")
    widths = [34, 14, 16, 14, 8]
    print_row(["model", "null core (8/doc)", "null all/fields", "total latency", "errors"], widths)
    print("-+-".join("-" * w for w in widths))
    for model, tot in totals.items():
        print_row(
            [model, str(tot["null_core"]), f"{tot['null_all']}/{tot['fields']}", f"{tot['latency']:.1f}s", str(tot["errors"])],
            widths,
        )


if __name__ == "__main__":
    asyncio.run(main())
