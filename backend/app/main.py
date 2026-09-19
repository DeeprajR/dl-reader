"""The FastAPI application: settings, startup checks, error handling and routes.

In the container it also serves the built frontend, so one process runs the whole app.
"""

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
# Local runs read backend/.env or a repo-root .env; in Docker the env comes from --env-file.
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(REPO_DIR / ".env")

# The imports below come after load_dotenv on purpose (hence "noqa: E402"), so the settings
# from .env are in place before any module of the app is imported.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from app.routes import chat, documents  # noqa: E402
import httpx  # noqa: E402

from app.services import ocr, rag, storage  # noqa: E402
from app.services.providers import ollama  # noqa: E402
from app.services.providers.base import OLLAMA_PREFIX, chat_model, is_local, llm_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("licence_reader")


class LocalhostLink(logging.Filter):
    """Print "Uvicorn running on http://localhost:7860" instead of the unopenable 0.0.0.0.

    The container must listen on 0.0.0.0 for `docker run -p` to reach it; only the message changes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and "0.0.0.0" in record.args:
            record.args = tuple("localhost" if arg == "0.0.0.0" else arg for arg in record.args)
        return True


logging.getLogger("uvicorn.error").addFilter(LocalhostLink())

# Where `npm run build` puts the frontend. It exists in the Docker image, and locally after a build.
FRONTEND_DIST = REPO_DIR / "frontend" / "dist"


def check_ollama(models: set[str]) -> None:
    """Warn (don't fail) if the local Ollama server or a configured model is missing."""
    url = ollama.ollama_url()
    try:
        # /api/tags lists the models Ollama has downloaded.
        tags = httpx.get(f"{url}/api/tags", timeout=3).json()
        installed = {m.get("name", "") for m in tags.get("models", [])}
    except Exception:
        logger.warning("Ollama is not reachable at %s. Start it with `ollama serve`, or set LLM_MODEL "
                       "to an OpenRouter model.", url)
        return
    for model in models:
        # Ollama names models "name:tag". A setting without a tag matches any tag of that name.
        if not any(name == model or name.split(":")[0] == model for name in installed):
            logger.warning("Ollama has no model '%s'. Run `ollama pull %s`.", model, model)


def startup_checks() -> None:
    """Fail fast on missing required config; warn about optional system tools."""
    models = (llm_model(), chat_model())
    # The key is required unless extraction and chat both run on a local Ollama model.
    if not all(is_local(m) for m in models) and not os.getenv("OPENROUTER_API_KEY"):
        message = (
            "OPENROUTER_API_KEY is not set. Copy backend/.env.example to .env in the repository "
            "root (or backend/.env) and add your OpenRouter API key."
        )
        logger.critical(message)
        raise RuntimeError(message)
    logger.info("Extraction model: %s | chat model: %s", *models)
    # A missing Ollama server or model is only a warning: it can be started after the app.
    if any(is_local(m) for m in models):
        check_ollama({m.removeprefix(OLLAMA_PREFIX) for m in models if is_local(m)})

    # Tesseract and poppler are optional at startup. Without them the app still runs, with less.
    try:
        logger.info("Tesseract %s found", ocr.tesseract_version())
    except Exception:
        logger.warning(
            "Tesseract was not found (set TESSERACT_CMD or add it to PATH). Extraction will still "
            "run, but nothing can be cross-checked: every field will be flagged for review."
        )

    poppler_dir = documents.poppler_path()
    if not (shutil.which("pdftoppm", path=poppler_dir) if poppler_dir else shutil.which("pdftoppm")):
        logger.warning("poppler (pdftoppm) was not found (set POPPLER_PATH or add it to PATH). PDF uploads will fail.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: check the settings, open the database, load the search model."""
    startup_checks()
    storage.init()
    rag.warm_up()
    yield


app = FastAPI(title="AI Driving Licence Reader", lifespan=lifespan)


# --- error handling -----------------------------------------------------------------------------
# Every error leaves the API in one shape, {"error": "message"}, so the frontend can always
# show `error` to the user.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Errors raised on purpose with HTTPException (400, 404, 502, ...)."""
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """A request body that does not match the model, for example a missing `question`.

    FastAPI reports a list of problems; they are joined into one readable sentence.
    """
    problems = []
    for err in exc.errors():
        # `loc` is the path to the bad value, e.g. ("body", "question"). The leading "body" is dropped.
        loc = ".".join(str(p) for p in err.get("loc", ())[1:])
        problems.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse({"error": "Invalid request: " + "; ".join(problems)}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Anything unexpected. The user gets a generic message. Only the error's type is logged, never
    its text, because that text could contain personal data from a licence.
    """
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse({"error": "An unexpected server error occurred."}, status_code=500)


app.include_router(documents.router)
app.include_router(chat.router)

# Serve the built frontend (Docker / production) when present. Registered last so /api wins.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
