"""Construction of authenticated Azure SDK clients.

Identity-first: every client uses DefaultAzureCredential (managed identity in
Azure, developer sign-in locally) unless an API key is explicitly configured.
SDK imports are local so that offline mode and CI do not pay for them at start-up.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import Settings

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


@lru_cache(maxsize=1)
def credential() -> Any:
    from azure.identity import DefaultAzureCredential

    # Interactive browser sign-in is excluded: a server must never block on a prompt.
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def openai_client(settings: Settings) -> Any:
    """Return an OpenAI-compatible client for chat and embeddings.

    Preference order:
    1. Foundry project endpoint -> AIProjectClient.get_openai_client() (Entra ID).
    2. Azure OpenAI endpoint with an Entra ID bearer-token provider.
    3. Azure OpenAI endpoint with an API key (fallback only).
    """
    if settings.project_endpoint:
        from azure.ai.projects import AIProjectClient

        project = AIProjectClient(endpoint=settings.project_endpoint, credential=credential())
        return project.get_openai_client()

    from openai import AzureOpenAI

    if settings.openai_api_key:
        return AzureOpenAI(
            azure_endpoint=settings.openai_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
        )
    from azure.identity import get_bearer_token_provider

    return AzureOpenAI(
        azure_endpoint=settings.openai_endpoint,
        azure_ad_token_provider=get_bearer_token_provider(credential(), COGNITIVE_SCOPE),
        api_version=settings.openai_api_version,
    )


def _search_credential(settings: Settings) -> Any:
    if settings.search_api_key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(settings.search_api_key)
    return credential()


def search_client(settings: Settings) -> Any:
    from azure.search.documents import SearchClient

    return SearchClient(settings.search_endpoint, settings.search_index, _search_credential(settings))


def search_index_client(settings: Settings) -> Any:
    from azure.search.documents.indexes import SearchIndexClient

    return SearchIndexClient(settings.search_endpoint, _search_credential(settings))


def content_safety_client(settings: Settings) -> Any:
    from azure.ai.contentsafety import ContentSafetyClient

    if settings.content_safety_key:
        from azure.core.credentials import AzureKeyCredential

        cred: Any = AzureKeyCredential(settings.content_safety_key)
    else:
        cred = credential()
    return ContentSafetyClient(settings.content_safety_endpoint, cred)
