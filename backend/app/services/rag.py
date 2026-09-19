"""Grounded "Ask the Document" chat: chunk, embed, retrieve, answer.

Retrieval is technically overkill for a one-page licence, but it keeps answers tied to cited
excerpts and scales unchanged to multi-page documents.
"""

import json
import logging
import threading
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from app.schemas import Box, ChatResponse, ChatSource, ExtractionResult, FieldValue
from app.services.extraction import iter_fields, normalize
from app.services.providers.base import ProviderError, chat_model
from app.services.providers.openrouter import complete, openrouter_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_CHARS = 200
TOP_K = 5
# Cosine distance above which a chunk is treated as unrelated to the question. If nothing is
# closer, the question is refused without calling the LLM (first grounding tier).
MAX_DISTANCE = 0.9
REFUSAL = "The document does not contain this information."

CHAT_SYSTEM_PROMPT = """
You answer questions about a driving licence using ONLY the provided document excerpts.
Rules:
1. If the answer is present, answer concisely and quote the exact supporting text.
2. If the answer is NOT in the excerpts, reply exactly: "The document does not contain this information." Do not use outside knowledge about licence formats.
3. Never speculate, estimate, or fill gaps.
""".strip()

_chroma_path = Path("chroma")
_client = None
_client_lock = threading.Lock()
_embedder = None
_embedder_lock = threading.Lock()
_doc_locks: dict[str, threading.RLock] = {}
_doc_locks_guard = threading.Lock()


def configure(path: Path | str) -> None:
    """Point the vector store at `path` (default ./chroma)."""
    global _chroma_path, _client
    with _client_lock:
        _chroma_path, _client = Path(path), None


def _chroma():
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(
                path=str(_chroma_path), settings=Settings(anonymized_telemetry=False)
            )
        return _client


def _embed(texts: list[str]) -> list[list[float]]:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer  # slow import; load on first use

            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True).tolist()


def _doc_lock(doc_id: str) -> threading.RLock:
    with _doc_locks_guard:
        return _doc_locks.setdefault(doc_id, threading.RLock())


def _collection_name(doc_id: str) -> str:
    return f"doc_{doc_id}"


# --- chunking -------------------------------------------------------------------------------


def chunk_ocr_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """~`size`-char chunks of whole OCR lines; consecutive chunks overlap by one line."""
    units: list[str] = []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if len(line) <= size:
            units.append(line)
            continue
        current = ""
        for word in line.split():  # an over-long line is split on word boundaries
            if current and len(current) + 1 + len(word) > size:
                units.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            units.append(current)

    chunks: list[str] = []
    i = 0
    while i < len(units):
        j, length = i, 0
        while j < len(units) and (j == i or length + len(units[j]) + 1 <= size):
            length += len(units[j]) + 1
            j += 1
        chunks.append("\n".join(units[i:j]))
        if j >= len(units):
            break
        i = j - 1 if j - 1 > i else j
    return chunks


def field_chunk(name: str, field: FieldValue) -> str:
    key = name.removeprefix("other_fields.")
    source = " ".join((field.source_text or "").split())
    return f"Field: {key} = {field.value} (source: '{source}')" if source else f"Field: {key} = {field.value}"


# --- indexing -------------------------------------------------------------------------------


def index_document(doc_id: str, result: ExtractionResult) -> None:
    """(Re)build the document's collection: OCR text chunks + one chunk per extracted field."""
    items: list[tuple[str, str, Box | None]] = [(c, "ocr_text", None) for c in chunk_ocr_text(result.ocr_text)]
    items += [
        (field_chunk(name, field), "extracted_fields", field.bbox)
        for name, field in iter_fields(result.data)
        if field.value is not None
    ]
    texts = [text for text, _, _ in items]
    embeddings = _embed(texts) if texts else []

    with _doc_lock(doc_id):
        client = _chroma()
        try:
            client.delete_collection(_collection_name(doc_id))  # idempotent re-extraction
        except NotFoundError:
            pass
        collection = client.create_collection(
            _collection_name(doc_id), configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
        if items:
            collection.add(
                ids=[str(i) for i in range(len(items))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {"origin": origin, "bbox": json.dumps(bbox.model_dump()) if bbox else ""}
                    for _, origin, bbox in items
                ],
            )
    logger.info("Indexed %d chunks for %s", len(items), doc_id)


def index_document_safely(doc_id: str, result: ExtractionResult) -> None:
    """Background-task wrapper: indexing failures are logged; chat re-indexes on demand."""
    try:
        index_document(doc_id, result)
    except Exception:
        logger.exception("Background indexing failed for %s", doc_id)


def ensure_indexed(doc_id: str, result: ExtractionResult) -> None:
    """Build the index if it is missing (e.g. chat arrives before the background task ran)."""
    with _doc_lock(doc_id):
        try:
            if _chroma().get_collection(_collection_name(doc_id)).count() > 0:
                return
        except NotFoundError:
            pass
        index_document(doc_id, result)


# --- retrieval + answer ---------------------------------------------------------------------


def retrieve(doc_id: str, question: str, k: int = TOP_K) -> list[dict]:
    """Top-k chunks as {text, origin, bbox, distance}, nearest first."""
    try:
        collection = _chroma().get_collection(_collection_name(doc_id))
    except NotFoundError:
        return []
    n = min(k, collection.count())
    if n == 0:
        return []
    res = collection.query(
        query_embeddings=_embed([question]), n_results=n, include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "text": text,
            "origin": meta["origin"],
            "bbox": Box(**json.loads(meta["bbox"])) if meta.get("bbox") else None,
            "distance": distance,
        }
        for text, meta, distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def is_refusal(answer: str) -> bool:
    return normalize(answer).startswith(normalize(REFUSAL))


async def _ask_llm(messages: list[dict]) -> str:
    client, model = openrouter_client(), chat_model()
    try:
        return await complete(client, model, messages)
    except ProviderError as e:
        if not e.retryable:
            raise
        logger.warning("Chat call to %s failed, retrying once: %s", model, e)
        return await complete(client, model, messages)


async def answer_question(question: str, chunks: list[dict]) -> ChatResponse:
    """Grounded answer from retrieved chunks, or the exact refusal string."""
    relevant = [c for c in chunks if c["distance"] <= MAX_DISTANCE]
    if not relevant:
        return ChatResponse(answer=REFUSAL, sources=[])

    excerpts = "\n\n".join(f"[{i}] ({c['origin']}) {c['text']}" for i, c in enumerate(relevant, start=1))
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpts:\n{excerpts}\n\nQuestion: {question}"},
    ]
    answer = (await _ask_llm(messages)).strip()
    if not answer or is_refusal(answer):
        return ChatResponse(answer=REFUSAL, sources=[])
    return ChatResponse(
        answer=answer,
        sources=[ChatSource(text=c["text"], origin=c["origin"], bbox=c["bbox"]) for c in relevant],
    )
