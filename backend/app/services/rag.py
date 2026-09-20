"""Grounded "Ask the Document" chat: chunk, embed, retrieve, answer.

Retrieval is technically overkill for a one-page licence, but it keeps answers tied to cited
excerpts and scales unchanged to multi-page documents.
"""

import json
import logging
import re
import threading
from datetime import date
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from app.schemas import Box, ChatResponse, ChatSource, ExtractionResult, FieldValue
from app.services import storage
from app.services.extraction import as_date, class_date_key, iter_fields, match_bbox, normalize
from app.services.providers import ollama
from app.services.providers.base import OLLAMA_PREFIX, ProviderError, chat_model, is_local
from app.services.providers.openrouter import complete, openrouter_client

logger = logging.getLogger(__name__)

# The model that turns a text into a list of numbers (an "embedding"). Texts with a similar
# meaning get similar numbers, which is how the chat finds the passages that fit a question.
# It is small (about 90 MB) and runs on this machine.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# The document text is split into passages ("chunks") of about this many characters.
CHUNK_CHARS = 200
# How many of the closest passages are given to the LLM for each question.
TOP_K = 5
# Cosine distance above which a chunk is treated as unrelated to the question. If nothing is
# closer, the question is refused without calling the LLM (first grounding tier).
MAX_DISTANCE = 0.9
# The exact sentence returned whenever the document cannot answer the question.
REFUSAL = "The document does not contain this information."

# The LLM's instructions for the chat. Rules 1-3 are the specification's, word for word.
# Rule 4 was added for date questions (see `date_facts`).
CHAT_SYSTEM_PROMPT = """
You answer questions about a driving licence using ONLY the provided document excerpts.
Rules:
1. If the answer is present, answer concisely and quote the exact supporting text.
2. If the answer is NOT in the excerpts, reply exactly: "The document does not contain this information." Do not use outside knowledge about licence formats.
3. Never speculate, estimate, or fill gaps.
4. An excerpt marked (calculated) holds date arithmetic the application worked out from the document's dates and today's date. For questions about days left, validity today, age or years held, use its numbers exactly as given, never calculate yourself, and also quote the printed date it is based on.
""".strip()

# Shared state, created on first use. Requests are handled by several threads, so every piece
# of shared state has a lock.
_chroma_path = Path("chroma")
_client = None
_client_lock = threading.Lock()
_embedder = None
_embedder_lock = threading.Lock()
# One lock per document, so rebuilding one document's index never blocks another document.
_doc_locks: dict[str, threading.RLock] = {}
_doc_locks_guard = threading.Lock()


def _chroma():
    """The ChromaDB client, opened on first use. It stores its data in the ./chroma folder."""
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(
                path=str(_chroma_path), settings=Settings(anonymized_telemetry=False)
            )
        return _client


def _embed(texts: list[str]) -> list[list[float]]:
    """The embedding of each text. The model is loaded on first use, because loading takes seconds."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer  # slow import; load on first use

            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
    # Normalised embeddings all have length 1, which is what cosine distance expects.
    return _embedder.encode(texts, normalize_embeddings=True).tolist()


def warm_up() -> None:
    """Load the embedding model in the background so the first chat is not slow."""

    def load():
        try:
            _embed(["warm-up"])
        except Exception:
            logger.exception("Could not load the embedding model %s", EMBEDDING_MODEL)

    threading.Thread(target=load, name="embedding-warm-up", daemon=True).start()


def _doc_lock(doc_id: str) -> threading.RLock:
    """The lock for one document, created the first time it is asked for."""
    with _doc_locks_guard:
        return _doc_locks.setdefault(doc_id, threading.RLock())


def _collection_name(doc_id: str) -> str:
    """The name of the document's ChromaDB collection. Each document has its own."""
    return f"doc_{doc_id}"


# --- chunking -------------------------------------------------------------------------------


def chunk_ocr_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """~`size`-char chunks of whole OCR lines; consecutive chunks overlap by one line."""
    # Step 1: cut the text into units. A unit is one OCR line, or a piece of a line that is longer
    # than a whole chunk.
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

    # Step 2: pack consecutive units into chunks of at most `size` characters. Each chunk starts
    # with the last unit of the previous one, so a fact split across two chunks is still whole in one.
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
        # Step back one unit for the overlap, unless the chunk held a single unit (that would loop forever).
        i = j - 1 if j - 1 > i else j
    return chunks


def field_chunk(name: str, field: FieldValue) -> str:
    """The search text for one form field, for example
    "Field: date_of_expiry = 2034-06-15 (source: 'Valid Till: 15-06-2034')".

    It holds both the cleaned value and the printed text, so either wording of a question finds it.
    """
    key = name.removeprefix("other_fields.")
    # An item's printed label is added, so a question in the card's own words ("S/D/W of") finds it.
    if field.label:
        key = f'{key} (printed as "{field.label}")'
    source = " ".join((field.source_text or "").split())
    return f"Field: {key} = {field.value} (source: '{source}')" if source else f"Field: {key} = {field.value}"


def class_validity_chunk(data) -> str | None:
    """One chunk with every vehicle class's dates, derived from the per-class fields.

    With many classes the per-class fields are many small chunks, and top-5 retrieval cannot
    return all of them for a question about every class; this chunk answers it in one piece.
    """
    # Collect the dates per class: {"lmv": {"date_of_issue": "...", "valid_till": "..."}}.
    classes: dict[str, dict[str, str]] = {}
    for key, field in data.other_fields.items():
        if (m := class_date_key(key)) and field.value:
            classes.setdefault(m["cls"], {})[m["kind"]] = field.value
    if not classes:
        return None
    parts = []
    for cls, dates in classes.items():
        label = cls.upper().replace("_", " ")
        issued, till = dates.get("date_of_issue", "not printed"), dates.get("valid_till", "not printed")
        parts.append(f"{label}: issued {issued}, valid till {till}")
    return "Vehicle class validity (all classes): " + "; ".join(parts)


# --- date answers -----------------------------------------------------------------------------
# "How many days until it expires?" needs today's date and arithmetic, and LLMs make arithmetic
# slips. So the app works the numbers out here and hands them to the LLM as one more excerpt.


def _days(n: int) -> str:
    """"1 day" or "N days"."""
    return "1 day" if n == 1 else f"{n} days"


def _years_between(start: date, end: date) -> int:
    """Whole years from `start` to `end` (an age, or how long a licence has been held)."""
    # Subtract one when this year's anniversary has not come round yet (comparing month and day).
    return end.year - start.year - ((end.month, end.day) < (start.month, start.day))


def _printed(field: FieldValue) -> str:
    """The text printed on the card for this date, so a calculated answer can still quote it."""
    source = " ".join((field.source_text or "").split())
    return f" (printed: '{source}')" if source else ""


def _validity(label: str, field: FieldValue, until: date, today: date) -> str:
    """One sentence saying when something expires, and whether that is ahead, today or in the past."""
    left = (until - today).days
    if left < 0:
        return f"{label} expired on {until}{_printed(field)}, {_days(-left)} ago (no longer valid today)."
    if left == 0:
        return f"{label} expires today, {until}{_printed(field)}."
    return f"{label} expires on {until}{_printed(field)}, {_days(left)} from today (still valid today)."


def date_facts(data, today: date) -> str | None:
    """Date arithmetic for the chat, done here so the LLM never has to calculate.

    Days until expiry (licence and each vehicle class), age and years held, worked out from the
    form's dates. It depends on today's date, so it is built per question and never indexed.
    """
    facts = []
    if expiry := as_date(data.date_of_expiry):
        facts.append(_validity("The licence", data.date_of_expiry, expiry, today))
    # Dates in the future are skipped: an age or a time held cannot be negative.
    if (birth := as_date(data.date_of_birth)) and birth <= today:
        age = _years_between(birth, today)
        facts.append(f"The holder was born on {birth}{_printed(data.date_of_birth)} and is {age} years old today.")
    if (issue := as_date(data.date_of_issue)) and issue <= today:
        held = (today - issue).days
        facts.append(
            f"The licence was issued on {issue}{_printed(data.date_of_issue)}, {_days(held)} ago "
            f"({_years_between(issue, today)} full years)."
        )
    # Each vehicle class has its own valid-till date.
    for key, field in data.other_fields.items():
        if (m := class_date_key(key)) and m["kind"] == "valid_till" and (till := as_date(field)):
            facts.append(_validity(f"Vehicle class {m['cls'].upper().replace('_', ' ')}", field, till, today))
    if not facts:
        return None
    return f"Calculated on {today} (today) from the licence dates: " + " ".join(facts)


# Questions about time: the only ones that get the calculated excerpt, so it is not listed as a
# source for "what is the licence number?".
_TIME_QUESTION = re.compile(
    r"\b(today|now|currently|days?|weeks?|months?|years?|old|age|aged|valid|validity|invalid|expir\w*"
    r"|lapsed?|left|remain\w*|until|till|still|long|renew\w*|overdue)\b",
    re.IGNORECASE,
)


def calculated_chunk(data, question: str, today: date) -> dict | None:
    """`date_facts` as a retrieved chunk, so it passes the same relevance gate as the others."""
    text = date_facts(data, today) if _TIME_QUESTION.search(question) else None
    if text is None:
        return None
    # The excerpt changes every day, so it cannot be stored in the index. Its distance to the
    # question is worked out here instead, in the same way ChromaDB does it for stored chunks.
    q, t = _embed([question, text])
    dot = sum(a * b for a, b in zip(q, t))
    norms = (sum(a * a for a in q) * sum(b * b for b in t)) ** 0.5
    distance = 1.0 - dot / norms if norms else 1.0  # cosine distance, as ChromaDB reports it
    # Expanding this source highlights the printed expiry date it is based on.
    return {"text": text, "origin": "calculated", "bbox": data.date_of_expiry.bbox, "distance": distance}


# --- indexing -------------------------------------------------------------------------------


def index_document(doc_id: str, result: ExtractionResult) -> None:
    """(Re)build the document's collection: OCR text chunks + one chunk per extracted field.

    Field chunks carry the field's bbox; OCR chunks are matched to word boxes best-effort
    (same matcher as the fields) so chat sources can be highlighted on the image.
    """
    words = storage.get_ocr_words(doc_id)
    # Every search item is (text, where it came from, where it is on the image).
    items: list[tuple[str, str, Box | None]] = [
        (chunk, "ocr_text", match_bbox(chunk, words)) for chunk in chunk_ocr_text(result.ocr_text)
    ]
    items += [
        (field_chunk(name, field), "extracted_fields", field.bbox)
        for name, field in iter_fields(result.data)
        if field.value is not None
    ]
    # Plus one summary of every vehicle class's dates, when the licence has any.
    if summary := class_validity_chunk(result.data):
        items.append((summary, "extracted_fields", None))
    texts = [text for text, _, _ in items]
    # Embedding is the slow part, so it is done before the lock is taken.
    embeddings = _embed(texts) if texts else []

    with _doc_lock(doc_id):
        client = _chroma()
        try:
            # Delete and recreate, so re-reading a document or saving edits never leaves old chunks behind.
            client.delete_collection(_collection_name(doc_id))  # idempotent re-extraction
        except NotFoundError:
            pass
        # `embedding_function=None`: the embeddings are supplied by this module, not computed by ChromaDB.
        collection = client.create_collection(
            _collection_name(doc_id), configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
        if items:
            collection.add(
                ids=[str(i) for i in range(len(items))],
                embeddings=embeddings,
                documents=texts,
                # ChromaDB metadata cannot hold objects, so the box is stored as JSON text ("" means no box).
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
    # Under the lock, so two requests never build the same index at once.
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
    query = _embed([question])
    # Wait for a rebuild in progress: mid-rebuild the collection is briefly missing or empty,
    # and a question asked right after extraction would be refused.
    with _doc_lock(doc_id):
        try:
            collection = _chroma().get_collection(_collection_name(doc_id))
        except NotFoundError:
            return []
        # Never ask for more results than the collection holds.
        n = min(k, collection.count())
        if n == 0:
            return []
        res = collection.query(query_embeddings=query, n_results=n, include=["documents", "metadatas", "distances"])
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
    """True when the LLM's answer is the refusal sentence, whatever its quotes, case or trailing text.

    The specification requires the exact sentence, so every variation is replaced by it.
    """
    return normalize(answer).startswith(normalize(REFUSAL))


async def _ask_llm(messages: list[dict]) -> str:
    """LLM_CHAT_MODEL (else LLM_MODEL); "ollama/..." stays on this machine, like extraction."""
    # Build `ask` for the configured model. Both providers take the same list of messages.
    model = chat_model()
    if is_local(model):
        base_url, name = ollama.ollama_url(), model.removeprefix(OLLAMA_PREFIX)

        def ask():
            """Send the messages to the local Ollama server."""
            return ollama.complete(base_url, name, messages)
    else:
        client = openrouter_client()

        def ask():
            """Send the messages to OpenRouter."""
            return complete(client, model, messages)

    try:
        return await ask()
    except ProviderError as e:
        # A timeout or a busy server is worth one more try. A wrong API key is not.
        if not e.retryable:
            raise
        logger.warning("Chat call to %s failed, retrying once: %s", model, e)
        return await ask()


async def answer_question(question: str, chunks: list[dict]) -> ChatResponse:
    """Grounded answer from retrieved chunks, or the exact refusal string."""
    # Checkpoint 1: is anything in the document related to the question at all? If not, refuse
    # right away, without calling the LLM.
    relevant = [c for c in chunks if c["distance"] <= MAX_DISTANCE]
    if not relevant:
        return ChatResponse(answer=REFUSAL, sources=[])

    # The LLM sees numbered excerpts, each labelled with where it came from, and then the question.
    excerpts = "\n\n".join(f"[{i}] ({c['origin']}) {c['text']}" for i, c in enumerate(relevant, start=1))
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpts:\n{excerpts}\n\nQuestion: {question}"},
    ]
    answer = (await _ask_llm(messages)).strip()
    # Checkpoint 2: the LLM itself found no answer in the excerpts.
    if not answer or is_refusal(answer):
        return ChatResponse(answer=REFUSAL, sources=[])
    return ChatResponse(
        answer=answer,
        sources=[ChatSource(text=c["text"], origin=c["origin"], bbox=c["bbox"]) for c in relevant],
    )
