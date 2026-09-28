"""Retrievers: offline BM25 and Azure AI Search hybrid (keyword + vector + semantic rerank)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from .models import Chunk, SearchHit
from .text import tokenize

SELECT_FIELDS = ["id", "doc_id", "title", "section", "content", "source"]
VECTOR_FIELD = "content_vector"
SEMANTIC_CONFIG = "policy-semantic"


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[SearchHit]: ...


class Embedder:
    """Azure OpenAI embeddings, batched because the indexer embeds many chunks at once."""

    def __init__(self, client: Any, deployment: str, dimensions: int | None = None, batch_size: int = 16):
        self._client, self._deployment = client, deployment
        self._dimensions, self._batch = dimensions, batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            kwargs: dict[str, Any] = {"model": self._deployment, "input": list(texts[start : start + self._batch])}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            # The API returns items with an index; sort to be safe rather than trust ordering.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors


class BM25Retriever:
    """Okapi BM25 over the local chunks.

    Chosen over TF-IDF cosine because length normalisation matters when chunk
    sizes vary, and because BM25 is the same lexical scorer Azure AI Search uses,
    so offline behaviour is a fair approximation of the keyword half of hybrid search.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75, min_score: float = 2.0):
        self.chunks = list(chunks)
        self.k1, self.b, self.min_score = k1, b, min_score
        # Title and section are indexed with the body because questions often name the policy.
        self._docs = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in self.chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)
        df = Counter(term for doc in self._docs for term in doc)
        n = len(self._docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_terms: list[str], index: int) -> float:
        doc, length, total = self._docs[index], self._lengths[index], 0.0
        for term in set(query_terms):
            tf = doc.get(term, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        terms = tokenize(query)
        scored = [SearchHit(c, self.score(terms, i)) for i, c in enumerate(self.chunks)]
        scored = [h for h in scored if h.score >= self.min_score]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]


class AzureSearchRetriever:
    """Hybrid query against Azure AI Search.

    One request carries both the BM25 text query and a vector query; the service
    fuses them with Reciprocal Rank Fusion, then (optionally) the semantic ranker
    re-scores the fused list. RRF scores are not comparable across queries, so the
    relevance cut-off is applied only to the semantic reranker score (0-4 scale).
    """

    def __init__(
        self,
        search_client: Any,
        embedder: Embedder,
        *,
        use_semantic_ranker: bool = True,
        min_reranker_score: float = 1.5,
    ):
        self._client, self._embedder = search_client, embedder
        self._semantic, self._min_rerank = use_semantic_ranker, min_reranker_score

    def build_query(self, query: str, top_k: int) -> dict[str, Any]:
        from azure.search.documents.models import QueryType, VectorizedQuery

        vector = self._embedder.embed([query])[0]
        # Ask the vector leg for more neighbours than we return so RRF has candidates to fuse.
        kwargs: dict[str, Any] = {
            "search_text": query,
            "vector_queries": [VectorizedQuery(vector=vector, k_nearest_neighbors=top_k * 3, fields=VECTOR_FIELD)],
            "select": SELECT_FIELDS,
            "top": top_k,
        }
        if self._semantic:
            kwargs["query_type"] = QueryType.SEMANTIC
            kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG
        return kwargs

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        hits = []
        for doc in self._client.search(**self.build_query(query, top_k)):
            reranker = doc.get("@search.reranker_score")
            if self._semantic and reranker is not None and reranker < self._min_rerank:
                continue
            chunk = Chunk(
                chunk_id=doc["id"],
                doc_id=doc.get("doc_id", ""),
                title=doc.get("title", ""),
                section=doc.get("section", ""),
                text=doc.get("content", ""),
                source=doc.get("source", ""),
            )
            hits.append(SearchHit(chunk, float(reranker if reranker is not None else doc.get("@search.score", 0.0))))
        return hits
