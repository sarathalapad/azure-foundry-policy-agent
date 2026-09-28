"""Build (or rebuild) the Azure AI Search index from the local policy documents.

Run with `python -m policyagent build-index`. The schema is defined in code so
that it is reviewed and versioned with the retriever that depends on it.
"""

from __future__ import annotations

import logging
from typing import Any

from . import azure_clients
from .config import Settings
from .corpus import load_corpus
from .models import Chunk
from .retrieval import SEMANTIC_CONFIG, VECTOR_FIELD, Embedder

log = logging.getLogger(__name__)
HNSW_CONFIG = "policy-hnsw"
VECTOR_PROFILE = "policy-vector-profile"


def build_index_schema(name: str, dimensions: int) -> Any:
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="title", analyzer_name="en.microsoft"),
        SearchableField(name="section", analyzer_name="en.microsoft"),
        SearchableField(name="content", analyzer_name="en.microsoft"),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    # HNSW with cosine distance: the right metric for OpenAI embeddings. m / ef values are
    # the service defaults, stated explicitly so a change is a visible diff.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_CONFIG,
                parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine"),
            )
        ],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG)],
    )
    semantic = SemanticSearch(
        default_configuration_name=SEMANTIC_CONFIG,
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="section"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="title")],
                ),
            )
        ],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search, semantic_search=semantic)


def to_documents(chunks: list[Chunk], vectors: list[list[float]]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "section": c.section,
            "content": c.text,
            "source": c.source,
            VECTOR_FIELD: v,
        }
        for c, v in zip(chunks, vectors, strict=True)
    ]


def build_index(settings: Settings) -> int:
    if not settings.azure_search_enabled:
        raise RuntimeError("Azure AI Search and Azure OpenAI must be configured to build the index.")
    chunks = load_corpus(settings.docs_dir)
    embedder = Embedder(
        azure_clients.openai_client(settings), settings.embedding_deployment, settings.embedding_dimensions
    )
    vectors = embedder.embed([f"{c.title}. {c.section}. {c.text}" for c in chunks])

    azure_clients.search_index_client(settings).create_or_update_index(
        build_index_schema(settings.search_index, settings.embedding_dimensions)
    )
    results = azure_clients.search_client(settings).upload_documents(documents=to_documents(chunks, vectors))
    failed = [r.key for r in results if not r.succeeded]
    if failed:
        raise RuntimeError(f"{len(failed)} documents failed to index: {failed[:5]}")
    log.info("Indexed %d chunks into %s", len(chunks), settings.search_index)
    return len(chunks)
