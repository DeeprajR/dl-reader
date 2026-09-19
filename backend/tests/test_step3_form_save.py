"""Step 3 - form save (PUT /data) and the frontend's contract with the API.

UI behaviour (staged progress, amber review styling, highlights) is verified in the browser;
these tests pin what the backend and build rely on.
"""

import pytest
from conftest import REPO_DIR, make_licence

pytestmark = [pytest.mark.phase1, pytest.mark.step3]

FRONTEND = REPO_DIR / "frontend"


def saved_data(client, doc_id):
    return client.get(f"/api/documents/{doc_id}/extract").json()["data"]


def test_save_persists_edits_across_reopen(client, extracted):
    data = saved_data(client, extracted)
    data["issuing_authority"]["value"] = "  RTO, Pune (verified)  "
    data["date_of_issue"]["value"] = "16/06/2019"  # normalised on save
    data["address"]["value"] = ""  # cleared -> null
    data["other_fields"]["blood_group"]["value"] = "B+"

    response = client.put(f"/api/documents/{extracted}/data", json=data)
    assert response.status_code == 200, response.text
    returned = response.json()
    assert returned["issuing_authority"]["value"] == "RTO, Pune (verified)"
    assert returned["date_of_issue"]["value"] == "2019-06-16"
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
    data = saved_data(client, extracted)
    data[field]["value"] = value
    response = client.put(f"/api/documents/{extracted}/data", json=data)
    assert response.status_code == 422
    assert fragment in response.json()["error"]
    assert saved_data(client, extracted)[field]["value"] != value  # nothing persisted


def test_save_rejects_bad_other_field_names_and_shapes(client, extracted):
    data = saved_data(client, extracted)
    data["other_fields"]["Bad Key!"] = data["other_fields"]["blood_group"]
    assert client.put(f"/api/documents/{extracted}/data", json=data).status_code == 422

    response = client.put(f"/api/documents/{extracted}/data", json={"full_name": "just a string"})
    assert response.status_code == 422
    assert response.json()["error"].startswith("Invalid request")


def test_save_requires_an_extraction(client, uploaded):
    response = client.put(f"/api/documents/{uploaded}/data", json=make_licence().model_dump())
    assert response.status_code == 404
    assert "not been extracted" in response.json()["error"]


# --- frontend contract -----------------------------------------------------------------------


def read(path):
    return (FRONTEND / path).read_text(encoding="utf-8")


def test_frontend_uses_relative_api_base_and_dev_proxy():
    assert "const API = '/api'" in read("src/api.js")  # same code in dev and in the container
    config = read("vite.config.js")
    assert "'/api'" in config and "8000" in config


def test_frontend_is_react_18_with_vite():
    import json

    package = json.loads(read("package.json"))
    assert package["dependencies"]["react"].lstrip("^~").startswith("18.")
    assert "vite" in package["devDependencies"]
    assert package["scripts"]["build"] == "vite build"


def test_frontend_implements_spec_views_and_texts():
    for view in ("DocumentListView", "UploadView", "DocumentWorkspace"):
        assert (FRONTEND / "src" / "views" / f"{view}.jsx").is_file()
    form = read("src/components/ExtractedForm.jsx")
    for text in ("Reading document…", "Extracting fields…", "Verifying…", "Please verify", "Source:", "Retry"):
        assert text in form + read("src/components/ErrorMessage.jsx"), text


@pytest.mark.skipif(not (FRONTEND / "dist" / "index.html").is_file(), reason="frontend not built (npm run build)")
def test_backend_serves_built_frontend_without_shadowing_api(client):
    page = client.get("/")
    assert page.status_code == 200 and "<div id=\"root\">" in page.text
    assert client.get("/api/documents").headers["content-type"].startswith("application/json")
