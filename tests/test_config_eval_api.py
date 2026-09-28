from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from policyagent.api import create_app
from policyagent.config import Settings
from policyagent.evaluation import Case, score_case, summarise
from policyagent.models import AgentResponse, Citation

AZURE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example-aoai.openai.azure.com",
    "AZURE_SEARCH_ENDPOINT": "https://example-search.search.windows.net",
    "AZURE_CONTENT_SAFETY_ENDPOINT": "https://example-cs.cognitiveservices.azure.com",
}


def test_config_selects_offline_when_env_missing():
    settings = Settings.from_env({})
    assert settings.mode == "offline"
    assert not (settings.azure_llm_enabled or settings.azure_search_enabled or settings.azure_safety_enabled)


def test_config_selects_azure_when_configured_and_offline_flag_overrides():
    assert Settings.from_env(AZURE_ENV).mode == "azure"
    assert Settings.from_env(AZURE_ENV, force_offline=True).mode == "offline"
    assert Settings.from_env({**AZURE_ENV, "POLICYAGENT_MODE": "offline"}).mode == "offline"


def test_config_search_requires_llm_for_embeddings():
    settings = Settings.from_env({"AZURE_SEARCH_ENDPOINT": "https://s.search.windows.net"})
    assert not settings.azure_search_enabled
    assert Settings.from_env({"AZURE_AI_PROJECT_ENDPOINT": "https://p", **AZURE_ENV}).azure_llm_enabled
    mixed = Settings.from_env({"AZURE_OPENAI_ENDPOINT": "https://o"}).mode
    assert mixed == "mixed(llm=azure,search=offline,safety=offline)"


def _citation(chunk_id, doc_id):
    return Citation(1, chunk_id, doc_id, "t", "s", "f.md")


def test_score_case_answered_correctly():
    case = Case("c", "q", "DOC-1", ["7 years", "destroyed"])
    response = AgentResponse(
        "q", "Kept 7 years [1].", "answered", [_citation("DOC-1-0", "DOC-1")], retrieved_ids=["DOC-1-0", "DOC-2-0"]
    )
    row = score_case(case, response)
    assert row["abstention_correct"] and row["citation_valid"] and row["cited_expected_doc"]
    assert row["keyword_recall"] == 0.5


def test_score_case_citation_not_retrieved_is_invalid_and_false_answer_penalised():
    case = Case("c", "q", None, [])
    response = AgentResponse("q", "x [1]", "answered", [_citation("DOC-9-0", "DOC-9")], retrieved_ids=["DOC-1-0"])
    row = score_case(case, response)
    assert row["abstention_correct"] is False and row["citation_valid"] is False
    assert row["keyword_recall"] is None


def test_summarise_ignores_not_applicable_values():
    rows = [
        {"abstention_correct": True, "keyword_recall": 1.0, "citation_valid": None},
        {"abstention_correct": False, "keyword_recall": 0.5, "citation_valid": True},
        {"safety_correct": True},
    ]
    assert summarise(rows) == {
        "abstention_correct": 0.5,
        "citation_valid": 1.0,
        "keyword_recall": 0.75,
        "safety_correct": 1.0,
    }


@pytest.fixture
def client(offline_agent):
    return TestClient(create_app(agent=offline_agent, settings=Settings.from_env({})))


def test_api_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "offline"


def test_api_ask_returns_citations(client):
    body = client.post("/ask", json={"question": "Which WCAG level must government websites meet?"}).json()
    assert body["outcome"] == "answered"
    assert body["citations"][0]["doc_id"] == "MDS-STD-004"


def test_api_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
