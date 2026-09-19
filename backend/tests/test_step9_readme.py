"""Step 9 (Phase 2) - README.md contains everything the spec requires."""

import pytest
from conftest import REPO_DIR

pytestmark = [pytest.mark.phase2, pytest.mark.step9]

REQUIRED = [
    # sections
    "technology stack",
    "architecture",
    "ai/llm approach",
    "model choice",
    "key technical decisions",
    "known limitations",
    "deployment",
    "ai development tools",
    # setup on every OS
    "brew install tesseract poppler",
    "apt install tesseract-ocr poppler-utils",
    "ub mannheim",
    "conda install -c conda-forge tesseract poppler",
    "tesseract_cmd",
    "poppler_path",
    # run commands
    "uvicorn",
    "npm run dev",
    "pytest",
    "docker build -t licence-reader .",
    "docker run -p 7860:7860 --env-file .env licence-reader",
    "openrouter_api_key",
    # model choice backed by the comparison
    "compare_models.py",
    "google/gemini-3.8-flash",
    "anthropic/claude-sonnet-5",
]


@pytest.fixture(scope="module")
def readme() -> str:
    return (REPO_DIR / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("phrase", REQUIRED)
def test_readme_mentions(readme, phrase):
    assert phrase in readme.lower()


def test_readme_has_both_mermaid_diagrams(readme):
    assert readme.count("```mermaid") >= 2  # pipeline + chat grounding tiers


def test_readme_has_no_unfilled_placeholders(readme):
    assert "<<" not in readme and "TODO" not in readme
