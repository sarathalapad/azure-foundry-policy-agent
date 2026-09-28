"""Wiring tests that use the real SDK client classes with the network replaced."""

from __future__ import annotations

import json

import pytest
from openai import AzureOpenAI

from policyagent import agent as agent_module
from policyagent import azure_clients
from policyagent.answering import AzureOpenAIAnswerer
from policyagent.config import Settings
from policyagent.retrieval import AzureSearchRetriever
from policyagent.safety import AzureContentSafetyScreen

from .conftest import fake_openai_client
from .test_azure_providers import HITS, FakeContentSafety, FakeSearchClient

hx = pytest.importorskip("httpx2")


def test_real_azure_openai_client_round_trip_with_entra_token():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return hx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Records are kept 7 years [1]."},
                    }
                ],
            },
        )

    client = AzureOpenAI(
        azure_endpoint="https://example.openai.azure.com",
        azure_ad_token_provider=lambda: "test-token",
        api_version="2024-10-21",
        http_client=hx.Client(transport=hx.MockTransport(handler)),
    )
    draft = AzureOpenAIAnswerer(client, "gpt-4o-mini").answer("How long are records kept?", HITS)

    assert draft.text == "Records are kept 7 years [1]." and draft.cited_numbers == [1]
    assert "/openai/deployments/gpt-4o-mini/chat/completions" in seen["url"]
    assert seen["auth"] == "Bearer test-token"
    assert seen["body"]["messages"][0]["role"] == "system"


def test_openai_client_uses_api_key_only_when_configured():
    settings = Settings.from_env(
        {"AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com", "AZURE_OPENAI_API_KEY": "placeholder"}
    )
    client = azure_clients.openai_client(settings)
    assert isinstance(client, AzureOpenAI) and client.api_key == "placeholder"


def test_build_agent_selects_azure_providers(monkeypatch):
    monkeypatch.setattr(azure_clients, "openai_client", lambda s: fake_openai_client())
    monkeypatch.setattr(azure_clients, "search_client", lambda s: FakeSearchClient([]))
    monkeypatch.setattr(azure_clients, "content_safety_client", lambda s: FakeContentSafety())
    settings = Settings.from_env(
        {
            "AZURE_OPENAI_ENDPOINT": "https://o",
            "AZURE_SEARCH_ENDPOINT": "https://s",
            "AZURE_CONTENT_SAFETY_ENDPOINT": "https://c",
        }
    )
    built = agent_module.build_agent(settings)
    assert isinstance(built.retriever, AzureSearchRetriever)
    assert isinstance(built.answerer, AzureOpenAIAnswerer)
    assert isinstance(built.safety, AzureContentSafetyScreen)
    assert built.mode == "azure"
