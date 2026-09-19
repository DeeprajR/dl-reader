"""POST /api/documents/{id}/chat - "Ask the document"; the grounding logic lives in services/rag.py."""

import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.routes.documents import get_document_or_404
from app.schemas import ChatRequest, ChatResponse, ExtractionResult
from app.services import rag, storage
from app.services.providers.base import ProviderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["chat"])

MAX_QUESTION_CHARS = 1000


def local_today(client_today: date | None) -> date:
    """The user's date. The server may run in another time zone (a container's clock is UTC), so
    the browser's date is used when it is plausible: within a day of the server's."""
    server_today = date.today()
    if client_today and abs((client_today - server_today).days) <= 1:
        return client_today
    return server_today


@router.post("/{doc_id}/chat", response_model=ChatResponse)
async def chat(doc_id: str, body: ChatRequest):
    """Answer one question from the document: validate, retrieve, then ask the LLM.

    503 = the search index failed, 502 = the LLM failed; a refusal is a normal 200 answer.
    """
    get_document_or_404(doc_id)
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Please enter a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(422, f"Questions are limited to {MAX_QUESTION_CHARS} characters.")

    raw = storage.get_extraction(doc_id)
    if raw is None:
        raise HTTPException(409, "Extract the document before asking questions about it.")
    result = ExtractionResult.model_validate_json(raw)

    try:
        await run_in_threadpool(rag.ensure_indexed, doc_id, result)
        chunks = await run_in_threadpool(rag.retrieve, doc_id, question)
        calculated = await run_in_threadpool(rag.calculated_chunk, result.data, question, local_today(body.today))
        if calculated:
            chunks = [*chunks, calculated]
    except Exception:
        logger.exception("Retrieval failed for %s", doc_id)
        raise HTTPException(503, "The document search index is unavailable right now. Please try again.")
    try:
        return await rag.answer_question(question, chunks)
    except ProviderError as e:
        raise HTTPException(502, f"The assistant could not answer: {e}")
