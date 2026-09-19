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

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
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

FRONTEND_DIST = REPO_DIR / "frontend" / "dist"


def check_ollama(models: set[str]) -> None:
    """Warn (don't fail) if the local Ollama server or a configured model is missing."""
    url = ollama.ollama_url()
    try:
        tags = httpx.get(f"{url}/api/tags", timeout=3).json()
        installed = {m.get("name", "") for m in tags.get("models", [])}
    except Exception:
        logger.warning("Ollama is not reachable at %s. Start it with `ollama serve`, or set LLM_MODEL "
                       "to an OpenRouter model.", url)
        return
    for model in models:
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
    if any(is_local(m) for m in models):
        check_ollama({m.removeprefix(OLLAMA_PREFIX) for m in models if is_local(m)})

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
    startup_checks()
    storage.init()
    rag.warm_up()
    yield


app = FastAPI(title="AI Driving Licence Reader", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())[1:])
        problems.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse({"error": "Invalid request: " + "; ".join(problems)}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse({"error": "An unexpected server error occurred."}, status_code=500)


app.include_router(documents.router)
app.include_router(chat.router)

# Serve the built frontend (Docker / production) when present. Registered last so /api wins.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
