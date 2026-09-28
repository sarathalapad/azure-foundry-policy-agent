"""Configuration from environment variables, and the offline/Azure mode decision.

Secrets never live in code or in the repository. In Azure the app runs with a
managed identity, so no keys are needed at all; API keys are read only as a
fallback for local experiments, and in production they should be App Service /
Container Apps Key Vault references rather than plain settings.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Azure AI Foundry project endpoint. When set, the OpenAI client is obtained
    # from the project (Entra ID only), which is the recommended Foundry path.
    project_endpoint: str | None = None
    # Direct Azure OpenAI / AI Services endpoint, used when no project endpoint is set.
    openai_endpoint: str | None = None
    openai_api_key: str | None = None
    openai_api_version: str = "2024-10-21"
    chat_deployment: str = "gpt-4o-mini"
    embedding_deployment: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    search_endpoint: str | None = None
    search_index: str = "policy-docs"
    search_api_key: str | None = None
    use_semantic_ranker: bool = True
    min_reranker_score: float = 1.5

    content_safety_endpoint: str | None = None
    content_safety_key: str | None = None
    # Azure Content Safety severities are 0/2/4/6; block at "medium" and above.
    block_severity: int = 4

    appinsights_connection_string: str | None = None
    docs_dir: Path = REPO_ROOT / "data" / "docs"
    top_k: int = 4
    force_offline: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, force_offline: bool = False) -> Settings:
        e = os.environ if env is None else env
        get = lambda name: (e.get(name) or "").strip() or None  # noqa: E731
        return cls(
            project_endpoint=get("AZURE_AI_PROJECT_ENDPOINT"),
            openai_endpoint=get("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=get("AZURE_OPENAI_API_KEY"),
            openai_api_version=get("AZURE_OPENAI_API_VERSION") or cls.openai_api_version,
            chat_deployment=get("AZURE_OPENAI_CHAT_DEPLOYMENT") or cls.chat_deployment,
            embedding_deployment=get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or cls.embedding_deployment,
            embedding_dimensions=int(get("AZURE_OPENAI_EMBEDDING_DIMENSIONS") or cls.embedding_dimensions),
            search_endpoint=get("AZURE_SEARCH_ENDPOINT"),
            search_index=get("AZURE_SEARCH_INDEX") or cls.search_index,
            search_api_key=get("AZURE_SEARCH_API_KEY"),
            use_semantic_ranker=_bool(get("AZURE_SEARCH_SEMANTIC_RANKER"), True),
            min_reranker_score=float(get("AZURE_SEARCH_MIN_RERANKER_SCORE") or cls.min_reranker_score),
            content_safety_endpoint=get("AZURE_CONTENT_SAFETY_ENDPOINT"),
            content_safety_key=get("AZURE_CONTENT_SAFETY_KEY"),
            block_severity=int(get("CONTENT_SAFETY_BLOCK_SEVERITY") or cls.block_severity),
            appinsights_connection_string=get("APPLICATIONINSIGHTS_CONNECTION_STRING"),
            docs_dir=Path(get("POLICY_DOCS_DIR") or cls.docs_dir),
            top_k=int(get("POLICYAGENT_TOP_K") or cls.top_k),
            force_offline=force_offline or (get("POLICYAGENT_MODE") or "").lower() == "offline",
        )

    @property
    def azure_llm_enabled(self) -> bool:
        return not self.force_offline and bool(self.project_endpoint or self.openai_endpoint)

    @property
    def azure_search_enabled(self) -> bool:
        # Hybrid search needs query embeddings, so it also needs the LLM endpoint.
        return self.azure_llm_enabled and bool(self.search_endpoint)

    @property
    def azure_safety_enabled(self) -> bool:
        return not self.force_offline and bool(self.content_safety_endpoint)

    @property
    def mode(self) -> str:
        """Human-readable summary, e.g. 'azure' or 'offline' or 'mixed(llm=azure,...)'."""
        flags = {
            "llm": self.azure_llm_enabled,
            "search": self.azure_search_enabled,
            "safety": self.azure_safety_enabled,
        }
        if all(flags.values()):
            return "azure"
        if not any(flags.values()):
            return "offline"
        return "mixed(" + ",".join(f"{k}={'azure' if v else 'offline'}" for k, v in flags.items()) + ")"
