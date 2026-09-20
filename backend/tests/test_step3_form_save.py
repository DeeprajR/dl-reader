"""Step 3 - form save (PUT /data) and the frontend's contract with the API.

UI behaviour (staged progress, amber review styling, highlights) is verified in the browser;
these tests pin what the backend and build rely on.
"""

import pytest
from conftest import REPO_DIR, make_licence

pytestmark = [pytest.mark.phase1, pytest.mark.step3]

FRONTEND = REPO_DIR / "frontend"


def saved_data(client, doc_id):
    """The stored form data, as the frontend gets it when a document is reopened."""
    return client.get(f"/api/documents/{doc_id}/extract").json()["data"]


def test_save_persists_edits_across_reopen(client, extracted):
    """Edits are tidied (trimmed, dates normalised, blanks to null), saved, and still there on reopen."""
    data = saved_data(client, extracted)
    data["issuing_authority"]["value"] = "  RTO, Pune (verified)  "
    data["date_of_issue"]["value"] = "16/06/2019"  # normalised on save
    data["address"]["value"] = ""  # cleared -> null
    data["other_fields"]["blood_group"]["value"] = "B+"

    response = client.put(f"/api/documents/{extracted}/data", json=data)
    assert response.status_code == 200, response.text
    returned = response.json()
    assert returned["issuing_authority"]["value"] == "RTO, Pune (verified)"
    assert returned["date_of_issue"]["value"] == "16-06-2019"
    assert returned["address"]["value"] is None

    reopened = saved_data(client, extracted)
    assert reopened == returned
    assert reopened["other_fields"]["blood_group"]["value"] == "B+"
    assert reopened["full_name"]["source_text"] == "JOHN DOE"  # provenance kept


@pytest.mark.parametrize(
    "field, value, fragment",
    [
        ("date_of_expiry", "not a date", "date_of_expiry"),
        ("date_of_birth", "31/02/2020", "date_of_birth"),
        ("full_name", "x" * 1001, "full_name"),
    ],
)
def test_save_rejects_invalid_values(client, extracted, field, value, fragment):
    """A bad date or an over-long value is a 422 that names the field, and nothing is saved."""
    data = saved_data(client, extracted)
    data[field]["value"] = value
    response = client.put(f"/api/documents/{extracted}/data", json=data)
    assert response.status_code == 422
    assert fragment in response.json()["error"]
    assert saved_data(client, extracted)[field]["value"] != value  # nothing persisted


def test_save_rejects_bad_other_field_names_and_shapes(client, extracted):
    """An unsafe field name, or a body of the wrong shape, is rejected with a 422."""
    data = saved_data(client, extracted)
    data["other_fields"]["Bad Key!"] = data["other_fields"]["blood_group"]
    assert client.put(f"/api/documents/{extracted}/data", json=data).status_code == 422

    response = client.put(f"/api/documents/{extracted}/data", json={"full_name": "just a string"})
    assert response.status_code == 422
    assert response.json()["error"].startswith("Invalid request")


def test_dates_are_day_first_everywhere(client, extracted):
    """DD-MM-YYYY is the one date format: stored, returned and shown as it is, with nothing converted in the browser."""
    data = saved_data(client, extracted)
    assert data["date_of_expiry"]["value"] == "15-06-2034"
    data["date_of_expiry"]["value"] = "2034-06-14"  # a year-first date is still read, and stored day first
    assert client.put(f"/api/documents/{extracted}/data", json=data).json()["date_of_expiry"]["value"] == "14-06-2034"

    form = read("src/components/ExtractedForm.jsx")
    assert "'DD-MM-YYYY'" in form and "YYYY-MM-DD" not in form + read("src/fields.js")


def test_save_keeps_printed_labels_of_other_fields_only(client, extracted):
    """A printed label is saved (tidied) with an item of other_fields, dropped from a core field, and limited in length."""
    data = saved_data(client, extracted)
    data["other_fields"]["blood_group"]["label"] = "  Blood   Group "
    data["full_name"]["label"] = "Name"
    saved = client.put(f"/api/documents/{extracted}/data", json=data).json()
    assert saved["other_fields"]["blood_group"]["label"] == "Blood Group"
    assert saved["full_name"]["label"] is None

    data["other_fields"]["blood_group"]["label"] = "x" * 65
    response = client.put(f"/api/documents/{extracted}/data", json=data)
    assert response.status_code == 422 and "label" in response.json()["error"]


def test_save_validates_per_class_dates(client, extracted):
    """Per-class dates such as lmv_valid_till are normalised and validated like the main dates."""
    data = saved_data(client, extracted)
    data["other_fields"]["lmv_valid_till"] = {**data["other_fields"]["blood_group"], "value": "15/06/2034"}
    assert client.put(f"/api/documents/{extracted}/data", json=data).json()["other_fields"]["lmv_valid_till"]["value"] == "15-06-2034"

    data["other_fields"]["lmv_valid_till"]["value"] = "someday"
    response = client.put(f"/api/documents/{extracted}/data", json=data)
    assert response.status_code == 422 and "lmv_valid_till" in response.json()["error"]


def test_save_requires_an_extraction(client, uploaded):
    """A form cannot be saved for a document that has not been read yet."""
    response = client.put(f"/api/documents/{uploaded}/data", json=make_licence().model_dump())
    assert response.status_code == 404
    assert "not been extracted" in response.json()["error"]


# --- frontend contract -----------------------------------------------------------------------


def read(path):
    """A frontend source file as text."""
    return (FRONTEND / path).read_text(encoding="utf-8")


def test_frontend_uses_relative_api_base_and_dev_proxy():
    """The frontend always calls /api, so the same code works with the dev server and in Docker."""
    assert "const API = '/api'" in read("src/api.js")  # same code in dev and in the container
    assert "res.status === 401" in read("src/api.js") and "/api/login" in read("src/api.js")  # signed out -> sign-in page
    config = read("vite.config.js")
    assert "'/api'" in config and "8000" in config


def test_frontend_is_react_18_with_vite():
    """The specification's frontend stack: React 18, built with Vite."""
    import json

    package = json.loads(read("package.json"))
    assert package["dependencies"]["react"].lstrip("^~").startswith("18.")
    assert "vite" in package["devDependencies"]
    assert package["scripts"]["build"] == "vite build"


def test_frontend_implements_spec_views_and_texts():
    """The three views exist, and the texts the specification requires are present."""
    for view in ("DocumentListView", "UploadView", "DocumentWorkspace"):
        assert (FRONTEND / "src" / "views" / f"{view}.jsx").is_file()
    form = read("src/components/ExtractedForm.jsx")
    for text in ("Reading document…", "Extracting fields…", "Verifying…", "Please verify", "Source:", "Retry"):
        assert text in form + read("src/components/ErrorMessage.jsx"), text


@pytest.mark.skipif(not (FRONTEND / "dist" / "index.html").is_file(), reason="frontend not built (npm run build)")
def test_backend_serves_built_frontend_without_shadowing_api(client):
    """In the container the backend serves the page at / while /api still answers JSON."""
    page = client.get("/")
    assert page.status_code == 200 and "<div id=\"root\">" in page.text
    assert client.get("/api/documents").headers["content-type"].startswith("application/json")


def test_form_has_exactly_the_nine_required_fields():
    """The form shows exactly the nine required fields, in order, under their required labels."""
    import re

    labels = re.findall(r"label: '([^']+)'", read("src/fields.js"))
    assert labels == [
        "Full Name",
        "Driving Licence Number",
        "Date of Birth",
        "Date of Issue",
        "Date of Expiry",
        "Address",
        "Vehicle/Class of Licence",
        "Issuing Authority",
        "Other relevant information",
    ]
    # An extra item is shown under its label as printed on the licence, not under a made-up name.
    assert "field?.label || humanize(key)" in read("src/fields.js")
    # Extra items are not rendered as their own inputs: they are lines of the ninth field.
    form = read("src/components/ExtractedForm.jsx")
    assert "<OtherField" in form and "Other details" not in form
