"""Step 4 - RAG indexing/retrieval and the grounded /chat endpoint."""

import pytest
from conftest import CANNED_OCR_TEXT

from app.schemas import ExtractionResult
from app.services import rag
from app.services.extraction import iter_fields

pytestmark = [pytest.mark.phase1, pytest.mark.step4]

REFUSAL = "The document does not contain this information."


def ask(client, doc_id, question):
    return client.post(f"/api/documents/{doc_id}/chat", json={"question": question})


def llm_replies(monkeypatch, reply):
    """Patch the chat completion; returns the list of message lists it received."""
    received = []

    async def fake_complete(client, model, messages):
        received.append(messages)
        return reply

    monkeypatch.setattr(rag, "complete", fake_complete)
    return received


# --- grounding / refusal ---------------------------------------------------------------------


def test_chat_refusal(client, extracted, monkeypatch):
    """Spec test 7: retrieval with no or irrelevant chunks -> exactly the refusal string."""
    monkeypatch.setattr(rag, "retrieve", lambda doc_id, question, k=5: [])
    response = ask(client, extracted, "What is this person's phone number?")
    assert response.status_code == 200
    assert response.json() == {"answer": REFUSAL, "sources": []}

    irrelevant = [{"text": "DRIVING LICENCE", "origin": "ocr_text", "bbox": None, "distance": 0.97}]
    monkeypatch.setattr(rag, "retrieve", lambda doc_id, question, k=5: irrelevant)
    response = ask(client, extracted, "Who won the football world cup?")
    assert response.json() == {"answer": REFUSAL, "sources": []}  # refused before any LLM call


@pytest.mark.parametrize(
    "reply",
    [
        REFUSAL,
        f'"{REFUSAL}"',
        f"{REFUSAL} The excerpts only show the name and address.",
        "the document does not contain this information",
    ],
)
def test_llm_refusals_are_normalised_to_the_exact_string(client, extracted, monkeypatch, reply):
    received = llm_replies(monkeypatch, reply)
    response = ask(client, extracted, "What is the licence holder's phone number?")
    assert received, "question should pass the relevance gate and reach the LLM"
    assert response.json() == {"answer": REFUSAL, "sources": []}


def test_grounded_answer_with_sources(client, extracted, monkeypatch):
    received = llm_replies(monkeypatch, 'The licence number is MH12 20190001234 ("DL No: MH12 20190001234").')
    response = ask(client, extracted, "What is the licence number?")
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("The licence number is MH12 20190001234")
    assert 1 <= len(body["sources"]) <= 5
    for source in body["sources"]:
        assert set(source) == {"text", "origin", "bbox"}
        assert source["origin"] in {"ocr_text", "extracted_fields"}
    assert any("MH12 20190001234" in s["text"] for s in body["sources"])

    system, user = received[0]
    assert system == {"role": "system", "content": rag.CHAT_SYSTEM_PROMPT}
    assert 'reply exactly: "The document does not contain this information."' in rag.CHAT_SYSTEM_PROMPT
    assert user["content"].rstrip().endswith("Question: What is the licence number?")
    assert "[1] (" in user["content"]  # labelled excerpts


def test_chat_uses_chat_model_setting(client, extracted, monkeypatch):
    models = []

    async def fake_complete(client_, model, messages):
        models.append(model)
        return "Answer."

    monkeypatch.setattr(rag, "complete", fake_complete)
    ask(client, extracted, "What is the licence number?")
    monkeypatch.setenv("LLM_CHAT_MODEL", "vendor/chat-model")
    ask(client, extracted, "What is the licence number?")
    assert models == ["test/fake-vision-model", "vendor/chat-model"]


def test_chat_provider_failure_is_clean_502(client, extracted, monkeypatch):
    from app.services.providers.base import ProviderError

    async def failing(client_, model, messages):
        raise ProviderError("upstream down")

    monkeypatch.setattr(rag, "complete", failing)
    response = ask(client, extracted, "What is the licence number?")
    assert response.status_code == 502
    assert "upstream down" in response.json()["error"]


# --- request validation ----------------------------------------------------------------------


def test_question_length_is_capped_at_1000(client, extracted, monkeypatch):
    llm_replies(monkeypatch, "ok")
    assert ask(client, extracted, "x" * 1000).status_code == 200
    response = ask(client, extracted, "x" * 1001)
    assert response.status_code == 422 and "1000" in response.json()["error"]


@pytest.mark.parametrize("body", [{"question": "   "}, {}, {"question": 5}])
def test_invalid_questions_are_rejected(client, extracted, body):
    response = client.post(f"/api/documents/{extracted}/chat", json=body)
    assert response.status_code == 422 and response.json()["error"]


def test_chat_requires_an_extraction(client, uploaded):
    response = ask(client, uploaded, "What is the licence number?")
    assert response.status_code == 409


# --- chunking + indexing ---------------------------------------------------------------------


def test_ocr_chunks_are_about_200_chars_with_overlap():
    lines = [f"line {i:02d} " + "x" * 50 for i in range(12)]
    chunks = rag.chunk_ocr_text("\n".join(lines))
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.splitlines()[-1] == current.splitlines()[0]  # one-line overlap
    assert {line for c in chunks for line in c.splitlines()} == set(lines)  # nothing lost

    long_line = " ".join(["word"] * 120)  # 599 chars, no line breaks
    assert all(len(c) <= 200 for c in rag.chunk_ocr_text(long_line))


def test_field_chunk_format():
    from conftest import fv

    assert rag.field_chunk("full_name", fv("JOHN DOE")) == "Field: full_name = JOHN DOE (source: 'JOHN DOE')"
    assert rag.field_chunk("other_fields.blood_group", fv("O+")) == "Field: blood_group = O+ (source: 'O+')"
    assert rag.field_chunk("address", fv("A, B", "A,\nB")) == "Field: address = A, B (source: 'A, B')"


def test_extraction_indexes_ocr_chunks_and_fields(client, extracted):
    result = ExtractionResult.model_validate(client.get(f"/api/documents/{extracted}/extract").json())
    stored = rag._chroma().get_collection(f"doc_{extracted}").get(include=["documents", "metadatas"])

    origins = [m["origin"] for m in stored["metadatas"]]
    assert origins.count("ocr_text") == len(rag.chunk_ocr_text(CANNED_OCR_TEXT))
    assert origins.count("extracted_fields") == sum(f.value is not None for _, f in iter_fields(result.data))
    assert "Field: licence_number = MH12 20190001234 (source: 'MH12 20190001234')" in stored["documents"]


def test_reindexing_is_idempotent_and_saved_edits_reach_chat(client, extracted):
    collection = lambda: rag._chroma().get_collection(f"doc_{extracted}")  # noqa: E731
    count = collection().count()

    client.post(f"/api/documents/{extracted}/extract")  # re-extraction
    assert collection().count() == count  # replaced, not duplicated

    data = client.get(f"/api/documents/{extracted}/extract").json()["data"]
    data["licence_number"]["value"] = "MH12 99999999999"
    client.put(f"/api/documents/{extracted}/data", json=data)
    documents = collection().get()["documents"]
    assert any("MH12 99999999999" in d for d in documents)
    assert collection().count() == count


def test_chat_builds_a_missing_index_on_demand(client, extracted, monkeypatch):
    rag._chroma().delete_collection(f"doc_{extracted}")
    llm_replies(monkeypatch, "The licence number is MH12 20190001234.")
    assert ask(client, extracted, "What is the licence number?").json()["sources"]


def test_retrieval_ranks_the_matching_chunk_first(client, extracted):
    top = rag.retrieve(extracted, "licence_number MH12 20190001234")[0]
    assert "MH12 20190001234" in top["text"]
    assert 0 <= top["distance"] <= rag.MAX_DISTANCE


def test_class_validity_summary_chunk():
    from conftest import fv, make_licence

    data = make_licence(other_fields={
        "lmv_date_of_issue": fv("2019-06-16"), "lmv_valid_till": fv("2034-06-15"),
        "lmv_tr_valid_till": fv("2021-03-11"), "blood_group": fv("O+"),
    })
    assert rag.class_validity_chunk(data) == (
        "Vehicle class validity (all classes): LMV: issued 2019-06-16, valid till 2034-06-15; "
        "LMV TR: issued not printed, valid till 2021-03-11"
    )
    assert rag.class_validity_chunk(make_licence()) is None  # no per-class fields, no summary


def test_summary_chunk_is_indexed_only_when_classes_have_dates(client, extracted):
    documents = rag._chroma().get_collection(f"doc_{extracted}").get()["documents"]
    assert not any(d.startswith("Vehicle class validity") for d in documents)  # canned licence has none

    data = client.get(f"/api/documents/{extracted}/extract").json()["data"]
    data["other_fields"]["mcwg_valid_till"] = {**data["other_fields"]["blood_group"], "value": "2035-06-04"}
    client.put(f"/api/documents/{extracted}/data", json=data)
    documents = rag._chroma().get_collection(f"doc_{extracted}").get()["documents"]
    assert "Vehicle class validity (all classes): MCWG: issued not printed, valid till 2035-06-04" in documents
