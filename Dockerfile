# One image: build the React frontend with Node, then run FastAPI (which also serves the built
# frontend at /) on slim Python with Tesseract and poppler.
#   docker build -t licence-reader .
#   docker run -p 7860:7860 --env-file .env licence-reader

# ---- Stage 1: frontend build --------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime --------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 app

WORKDIR /app/backend
COPY backend/requirements.txt ./
# CPU-only PyTorch first, so sentence-transformers does not pull ~2 GB of CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install -r requirements.txt

# Bake the embedding model into the image: no download on first chat, works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY backend/ ./
COPY --from=frontend /app/frontend/dist /app/frontend/dist
RUN mkdir -p /app/backend/data /app/backend/chroma && chown -R app:app /app

USER app
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
