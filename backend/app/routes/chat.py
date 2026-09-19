from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.routes.documents import get_document_or_404
from app.schemas import ChatRequest, ChatResponse, ExtractionResult
from app.services import rag, storage
from app.services.providers.base import ProviderError

router = APIRouter(prefix="/api/documents", tags=["chat"])

MAX_QUESTION_CHARS = 1000


@router.post("/{doc_id}/chat", response_model=ChatResponse)
async def chat(doc_id: str, body: ChatRequest):
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

    await run_in_threadpool(rag.ensure_indexed, doc_id, result)
    chunks = await run_in_threadpool(rag.retrieve, doc_id, question)
    try:
        return await rag.answer_question(question, chunks)
    except ProviderError as e:
        raise HTTPException(502, f"The assistant could not answer: {e}")
