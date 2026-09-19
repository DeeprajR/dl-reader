"""Step 2 - schemas, merge/confidence, provider + factory, /extract, persisted GET, compare script."""

import asyncio
import base64
import copy
import importlib.util
import io
import json
import time

import pytest
from conftest import (
    BACKEND_DIR,
    CANNED_OCR_TEXT,
    CANNED_WORDS,
    FakeProvider,
    fv,
    image_bytes,
    make_licence,
    upload,
)
from PIL import Image

from app.schemas import CORE_FIELDS, Box, ExtractionResult, FieldValue
from app.services import ocr
from app.services.extraction import LOW_OCR_WARNING, iter_fields, merge, normalize_date
from app.services.providers import base, openrouter
from app.services.providers.base import (
    EXTRACTION_SYSTEM_PROMPT,
    JSON_RETRY_PROMPT,
    InvalidJSONError,
    ProviderError,
    parse_licence_json,
)

pytestmark = [pytest.mark.phase1, pytest.mark.step2]


def confidences(data):
    return {name: field.confidence for name, field in iter_fields(data)}


# --- dates -----------------------------------------------------------------------------------


def test_normalize_dates():
    """Spec test 1."""
    for printed in ("15/01/2020", "15-01-2020", "15 JAN 2020"):
        assert normalize_date(printed) == "2020-01-15"
    assert normalize_date("garbage") is None


@pytest.mark.parametrize(
    "printed, iso",
    [
        ("2020-01-15", "2020-01-15"),
        ("15.01.2020", "2020-01-15"),
        ("15-Jan-2020", "2020-01-15"),
        ("15 January 2020", "2020-01-15"),
        ("January 15, 2020", "2020-01-15"),
        ("15/01/20", "2020-01-15"),
        ("12-08-1990", "1990-08-12"),
        ("31/02/2020", None),  # impossible date
        ("15 MAYBE 2020", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_date_formats(printed, iso):
    assert normalize_date(printed) == iso


# --- merge / confidence ----------------------------------------------------------------------


def test_confidence_merge():
    """Spec test 2: matching source_text -> high; non-matching -> review; OCR < 20 chars -> all review + warning."""
    data = make_licence(licence_number=fv("XY99 00000000000"))  # not printed in the OCR text
    merged, warnings = merge(data, CANNED_OCR_TEXT)
    assert merged.full_name.confidence == "high"
    assert merged.licence_number.confidence == "review"
    assert warnings == []

    merged, warnings = merge(make_licence(), "too little")
    assert set(confidences(merged).values()) == {"review"}
    assert LOW_OCR_WARNING in warnings


def test_merge_null_values_are_review():
    merged, _ = merge(make_licence(issuing_authority=fv(None)), CANNED_OCR_TEXT)
    assert merged.issuing_authority.confidence == "review"
    assert merged.issuing_authority.value is None


def test_merge_tolerates_small_ocr_errors():
    ocr_text = CANNED_OCR_TEXT.replace("JOHN DOE", "J0HN DOE")  # one OCR misread
    merged, _ = merge(make_licence(), ocr_text)
    assert merged.full_name.confidence == "high"


def test_merge_matches_dates_by_digits():
    ocr_text = CANNED_OCR_TEXT.replace("16-06-2019", "16/06 /2019")  # different separators
    merged, _ = merge(make_licence(), ocr_text)
    assert merged.date_of_issue.confidence == "high"


def test_merge_normalizes_date_values_and_warns_on_garbage():
    data = make_licence(date_of_issue=fv("16-06-2019"), date_of_birth=fv("sometime", "sometime"))
    merged, warnings = merge(data, CANNED_OCR_TEXT)
    assert merged.date_of_issue.value == "2019-06-16"
    assert merged.date_of_issue.source_text == "16-06-2019"  # printed form kept
    assert any("date_of_birth" in w for w in warnings)


def test_merge_does_not_mutate_input():
    data = make_licence()
    merge(data, CANNED_OCR_TEXT)
    assert data.full_name.confidence == "review"


# --- schema ----------------------------------------------------------------------------------


def test_schema_roundtrip():
    """Spec test 4: ExtractionResult serializes/deserializes losslessly."""
    data = make_licence()
    data.full_name = FieldValue(value="JOHN DOE", source_text="JOHN DOE", confidence="high", bbox=Box(x=1, y=2, w=3, h=4))
    data.other_fields["state"] = FieldValue(value=None, source_text=None, confidence="review", bbox=None, page=1)
    result = ExtractionResult(doc_id="d1", data=data, ocr_text="line 1\nline 2 “quoted”", warnings=["w1"])

    restored = ExtractionResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.model_dump() == json.loads(result.model_dump_json())


# --- parsing model replies -------------------------------------------------------------------


def licence_json(**overrides) -> str:
    obj = {name: {"value": field.value, "source_text": field.source_text} for name, field in iter_fields(make_licence())
           if not name.startswith("other_fields.")}
    obj["other_fields"] = {"blood_group": {"value": "O+", "source_text": "O+"}}
    obj.update(overrides)
    return json.dumps(obj)


def test_parse_accepts_fences_and_raw_newlines():
    reply = "```json\n" + licence_json().replace("\\n", "\n") + "\n```"
    data = parse_licence_json(reply)
    assert data.address.source_text == "12 High Street,\nPune"
    assert data.full_name.confidence == "review" and data.full_name.bbox is None


def test_parse_normalizes_loose_shapes():
    reply = licence_json(
        vehicle_classes=["LMV", "MCWG"],  # bare list instead of {value, source_text}
        issuing_authority={"value": None, "source_text": "ignored"},
        other_fields={"Blood Group": {"value": "O+", "source_text": "O+"}, "full_name": "dup", "State": "Pune"},
    )
    data = parse_licence_json(reply)
    assert data.vehicle_classes.value == "LMV, MCWG"
    assert data.issuing_authority.source_text is None  # no source without a value
    assert set(data.other_fields) == {"blood_group", "state"}  # snake_case; core-name clash dropped
    assert data.other_fields["state"].value == "Pune"


def test_parse_missing_fields_become_null():
    data = parse_licence_json('{"full_name": {"value": "A", "source_text": "A"}}')
    assert data.licence_number.value is None and data.other_fields == {}


@pytest.mark.parametrize("reply", ["", "Sorry, I cannot help.", "[1, 2]", "{not json}"])
def test_parse_rejects_non_json(reply):
    with pytest.raises(InvalidJSONError):
        parse_licence_json(reply)


# --- OpenRouter provider (completion mocked) --------------------------------------------------


def scripted_complete(monkeypatch, *replies):
    """Patch the completion call to return/raise `replies` in order; returns the call log."""
    calls = []

    async def fake_complete(client, model, messages):
        calls.append(copy.deepcopy(messages))
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(openrouter, "complete", fake_complete)
    return calls


def extract(model="test/model"):
    return asyncio.run(openrouter.OpenRouterProvider(model).extract(image_bytes((40, 30)), "image/png"))


def test_provider_request_uses_verbatim_prompt_and_image(monkeypatch):
    calls = scripted_complete(monkeypatch, licence_json())
    data = extract()
    assert data.licence_number.value == "MH12 20190001234"

    system, user = calls[0]
    assert system == {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}
    assert "A wrong licence number is worse than a null." in EXTRACTION_SYSTEM_PROMPT
    text_part, image_part = user["content"]
    for name in (*CORE_FIELDS, "other_fields"):
        assert f'"{name}"' in text_part["text"]  # schema shape in the user message
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == image_bytes((40, 30))


def test_provider_retries_once_on_invalid_json(monkeypatch):
    calls = scripted_complete(monkeypatch, "not json", licence_json())
    assert extract().full_name.value == "JOHN DOE"
    assert len(calls) == 2
    assert calls[1][-2] == {"role": "assistant", "content": "not json"}
    assert calls[1][-1] == {"role": "user", "content": JSON_RETRY_PROMPT}


def test_provider_gives_up_after_one_retry(monkeypatch):
    calls = scripted_complete(monkeypatch, "not json", "still not json", licence_json())
    with pytest.raises(ProviderError, match="valid JSON"):
        extract()
    assert len(calls) == 2


def test_provider_retries_transient_failures_once(monkeypatch):
    calls = scripted_complete(monkeypatch, ProviderError("timeout", retryable=True), licence_json())
    assert extract().full_name.value == "JOHN DOE"
    assert len(calls) == 2


def test_provider_does_not_retry_permanent_failures(monkeypatch):
    calls = scripted_complete(monkeypatch, ProviderError("bad key"), licence_json())
    with pytest.raises(ProviderError, match="bad key"):
        extract()
    assert len(calls) == 1


def test_client_config(monkeypatch):
    client = openrouter.openrouter_client()
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert client.timeout == 60
    assert client.max_retries == 0  # the one retry is ours, not the SDK's

    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        openrouter.openrouter_client()


# --- provider factory ------------------------------------------------------------------------


def test_factory_uses_llm_model_or_explicit_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "vendor/from-env")
    provider = base.get_provider()
    assert isinstance(provider, openrouter.OpenRouterProvider) and provider.model == "vendor/from-env"
    assert base.get_provider("  vendor/explicit ").model == "vendor/explicit"


def test_factory_defaults(monkeypatch):
    monkeypatch.delenv("LLM_MODEL")
    assert base.llm_model() == "google/gemini-3.8-flash"  # chosen via compare_models.py
    assert base.chat_model() == "google/gemini-3.8-flash"
    monkeypatch.setenv("LLM_CHAT_MODEL", "vendor/chat")
    assert base.chat_model() == "vendor/chat"


# --- /extract endpoint -----------------------------------------------------------------------


def test_extract_endpoint(client, uploaded, fake_provider, fake_ocr):
    """Spec test 6: FakeProvider + tiny generated image -> 200 and a valid ExtractionResult."""
    response = client.post(f"/api/documents/{uploaded}/extract")
    assert response.status_code == 200, response.text
    result = ExtractionResult.model_validate(response.json())
    assert result.doc_id == uploaded
    assert result.ocr_text == CANNED_OCR_TEXT
    assert result.warnings == []
    assert set(confidences(result.data).values()) == {"high"}
    assert result.data.date_of_birth.value == "1990-08-12"
    assert len(fake_provider.calls) == 1 and fake_provider.calls[0][1] == "image/png"


def test_extraction_is_persisted_and_reopened_without_rerun(client, uploaded, fake_provider, fake_ocr):
    assert client.get(f"/api/documents/{uploaded}/extract").status_code == 404  # never extracted
    posted = client.post(f"/api/documents/{uploaded}/extract").json()

    assert client.get(f"/api/documents/{uploaded}/extract").json() == posted
    assert len(fake_provider.calls) == 1
    assert client.get("/api/documents").json()[0]["has_extraction"] is True


def test_extract_runs_ocr_and_llm_in_parallel(client, uploaded, monkeypatch):
    def slow_ocr(path):
        time.sleep(0.6)
        return ocr.OcrResult(text=CANNED_OCR_TEXT, words=CANNED_WORDS)

    class SlowProvider(FakeProvider):
        async def extract(self, image_bytes, media_type):
            await asyncio.sleep(0.6)
            return make_licence()

    monkeypatch.setattr(ocr, "run_ocr", slow_ocr)
    monkeypatch.setattr(base, "get_provider", lambda model=None: SlowProvider())
    start = time.perf_counter()
    assert client.post(f"/api/documents/{uploaded}/extract").status_code == 200
    assert time.perf_counter() - start < 1.1  # sequential would take >= 1.2s


def test_extract_provider_failure_is_clean_502(client, uploaded, monkeypatch, fake_ocr):
    monkeypatch.setattr(base, "get_provider", lambda model=None: FakeProvider(error=ProviderError("model timed out")))
    response = client.post(f"/api/documents/{uploaded}/extract")
    assert response.status_code == 502
    assert response.json() == {"error": "Extraction failed: model timed out"}
    assert client.get(f"/api/documents/{uploaded}/extract").status_code == 404  # nothing persisted


def test_extract_degrades_when_ocr_fails(client, uploaded, fake_provider, monkeypatch):
    def broken_ocr(path):
        raise RuntimeError("tesseract missing")

    monkeypatch.setattr(ocr, "run_ocr", broken_ocr)
    result = client.post(f"/api/documents/{uploaded}/extract").json()
    assert result["data"]["full_name"]["value"] == "JOHN DOE"
    assert set(confidences(ExtractionResult.model_validate(result).data).values()) == {"review"}
    assert any("OCR could not run" in w for w in result["warnings"])
    assert LOW_OCR_WARNING in result["warnings"]


def test_extract_warns_when_no_licence_fields_found(client, uploaded, monkeypatch, fake_ocr):
    empty = make_licence(**{name: fv(None) for name in CORE_FIELDS}, other_fields={})
    monkeypatch.setattr(base, "get_provider", lambda model=None: FakeProvider(data=empty))
    warnings = client.post(f"/api/documents/{uploaded}/extract").json()["warnings"]
    assert any("No driving licence fields" in w for w in warnings)


def test_large_images_are_downscaled_for_the_llm_only(client, fake_provider, fake_ocr):
    doc_id = upload(client, image_bytes(size=(3000, 1500))).json()["doc_id"]
    client.post(f"/api/documents/{doc_id}/extract")

    sent, media_type = fake_provider.calls[0]
    assert media_type == "image/jpeg"
    assert max(Image.open(io.BytesIO(sent)).size) == 2048
    assert client.get(f"/api/documents/{doc_id}/meta").json() == {"width": 3000, "height": 1500}


# --- scripts/compare_models.py ---------------------------------------------------------------


def test_compare_models_prints_diff_table(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("compare_models", BACKEND_DIR / "scripts" / "compare_models.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    (tmp_path / "sample.png").write_bytes(image_bytes((200, 120)))
    (tmp_path / "notes.txt").write_text("ignored")
    providers = {
        "model/a": FakeProvider(make_licence()),
        "model/b": FakeProvider(make_licence(full_name=fv("JANE DOE"), address=fv(None))),
    }
    monkeypatch.setattr(script, "SAMPLES_DIR", tmp_path)
    monkeypatch.setattr(script, "get_provider", lambda model=None: providers[model])
    monkeypatch.setenv("LLM_MODEL", "model/a")
    monkeypatch.setenv("LLM_MODEL_ALT", "model/b")

    asyncio.run(script.main())
    out = capsys.readouterr().out
    rows = {line.split("|")[0].strip(): line for line in out.splitlines() if "|" in line}
    assert "agree?" in rows["field"]
    assert rows["full_name"].rstrip().endswith("NO")
    assert rows["licence_number"].rstrip().endswith("yes")
    assert "(null)" in rows["address"]
    assert rows["model/a"].split("|")[1].strip() == "0"  # null core fields
    assert rows["model/b"].split("|")[1].strip() == "1"
    assert all(len(p.calls) == 1 for p in providers.values())  # same sample through both models


def test_prompt_names_date_labels_and_asks_for_them_in_the_source():
    from app.services.providers.base import EXTRACTION_USER_PROMPT

    for label in ("DOI / Date of Issue", "Valid Till / Validity", "DOB / Date of Birth"):
        assert label in EXTRACTION_USER_PROMPT
    assert "source_text is the printed label together with the date" in EXTRACTION_USER_PROMPT


def test_prompt_asks_for_separate_per_class_dates_only_when_printed():
    from app.services.providers.base import EXTRACTION_USER_PROMPT

    assert '"<class>_date_of_issue" and "<class>_valid_till"' in EXTRACTION_USER_PROMPT
    assert "table row exactly as printed" in EXTRACTION_USER_PROMPT  # locates the right row
    assert "Omit them if no such table is printed" in EXTRACTION_USER_PROMPT
    assert "vehicle_class_validity" not in EXTRACTION_USER_PROMPT


@pytest.mark.parametrize(
    "text, iso",
    [
        ("DOI: 16-06-2019", "2019-06-16"),
        ("Valid Till: 15-06-2034 (NT)", "2034-06-15"),
        ("Date of Issue : 10-04-2018", "2018-04-10"),
        (":12-08-1990", "1990-08-12"),
        ("Issued on 5 Jan 2021", "2021-01-05"),
        ("no date here", None),
        (None, None),
    ],
)
def test_find_date_reads_dates_inside_labelled_text(text, iso):
    from app.services.extraction import find_date

    assert find_date(text) == iso


def test_merge_accepts_labelled_date_sources():
    data = make_licence(
        date_of_issue=fv("2019-06-16", "DOI : 16-06-2019"),
        date_of_expiry=fv("not a date", "Valid Till : 15-06-2034"),  # value recovered from the source
    )
    merged, warnings = merge(data, CANNED_OCR_TEXT)
    assert merged.date_of_issue.confidence == "high"
    assert merged.date_of_expiry.value == "2034-06-15"
    assert warnings == []


# --- date order sanity -----------------------------------------------------------------------


def test_swapped_issue_and_expiry_are_flagged():
    data = make_licence(date_of_issue=fv("2034-06-15", "15-06-2034"), date_of_expiry=fv("2019-06-16", "16-06-2019"))
    merged, warnings = merge(data, CANNED_OCR_TEXT)
    assert merged.date_of_issue.confidence == merged.date_of_expiry.confidence == "review"
    assert any("Were they swapped?" in w for w in warnings)
    assert merged.full_name.confidence == "high"  # other fields unaffected


def test_birth_after_issue_and_future_issue_are_flagged():
    data = make_licence(date_of_birth=fv("2020-01-01", "01-01-2020"))
    merged, warnings = merge(data, CANNED_OCR_TEXT)
    assert merged.date_of_birth.confidence == merged.date_of_issue.confidence == "review"
    assert any("not before the date of issue" in w for w in warnings)

    merged, warnings = merge(make_licence(date_of_issue=fv("2099-01-01")), CANNED_OCR_TEXT)
    assert any("in the future" in w for w in warnings)
    assert merged.date_of_issue.confidence == "review"


def test_per_class_dates_are_normalised_and_order_checked():
    other = {
        "lmv_date_of_issue": fv("16-06-2019", "LMV 16-06-2019 15-06-2034"),
        "lmv_valid_till": fv("2034-06-15", "LMV 16-06-2019 15-06-2034"),
        "mcwg_date_of_issue": fv("2034-06-15", "MCWG 16-06-2019 15-06-2034"),  # swapped
        "mcwg_valid_till": fv("2019-06-16", "MCWG 16-06-2019 15-06-2034"),
    }
    merged, warnings = merge(make_licence(other_fields=other), CANNED_OCR_TEXT)
    assert merged.other_fields["lmv_date_of_issue"].value == "2019-06-16"
    assert any(w.startswith("MCWG date of issue") and "swapped" in w for w in warnings)
    assert not any(w.startswith("LMV") for w in warnings)
    assert merged.other_fields["mcwg_date_of_issue"].confidence == "review"
