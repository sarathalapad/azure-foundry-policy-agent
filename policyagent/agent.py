"""The policy agent: screen -> retrieve -> answer -> validate citations -> screen.

The agent owns the grounding contract. Whatever the answerer returns, the agent
drops citation markers that do not point at a retrieved source and refuses to
return an answer with no valid citation at all. That rule lives here, not in the
prompt, because a prompt is a request and this is an enforcement point.
"""

from __future__ import annotations

import re

from .answering import ABSTAIN_MESSAGE, Answerer, AzureOpenAIAnswerer, ExtractiveAnswerer
from .config import Settings
from .models import AgentResponse, Citation, SearchHit
from .retrieval import AzureSearchRetriever, BM25Retriever, Embedder, Retriever
from .safety import AzureContentSafetyScreen, RegexSafetyScreen, SafetyScreen
from .telemetry import tracer

BLOCKED_MESSAGE = "I can't help with that request. Please rephrase your question about government policy."
MAX_QUESTION_CHARS = 2000


class PolicyAgent:
    def __init__(
        self,
        retriever: Retriever,
        answerer: Answerer,
        safety: SafetyScreen,
        *,
        top_k: int = 4,
        mode: str = "offline",
    ):
        self.retriever, self.answerer, self.safety = retriever, answerer, safety
        self.top_k, self.mode = top_k, mode

    def _response(self, question: str, answer: str, outcome: str, **kw) -> AgentResponse:
        return AgentResponse(question=question, answer=answer, outcome=outcome, mode=self.mode, **kw)  # type: ignore[arg-type]

    def ask(self, question: str) -> AgentResponse:
        question = question.strip()[:MAX_QUESTION_CHARS]
        with tracer.start_as_current_span("policyagent.ask") as span:
            span.set_attribute("policyagent.mode", self.mode)
            span.set_attribute("policyagent.question_chars", len(question))
            response = self._ask(question)
            span.set_attribute("policyagent.outcome", response.outcome)
            span.set_attribute("policyagent.citations", len(response.citations))
            span.set_attribute("policyagent.degraded", response.degraded)
            return response

    def _ask(self, question: str) -> AgentResponse:
        with tracer.start_as_current_span("safety.input"):
            verdict = self.safety.check_input(question)
        if not verdict.allowed:
            return self._response(question, BLOCKED_MESSAGE, "blocked", reasons=list(verdict.reasons))

        with tracer.start_as_current_span("retrieve") as span:
            hits = self.retriever.search(question, self.top_k)
            span.set_attribute("retrieve.hits", len(hits))
        retrieved_ids = [h.chunk.chunk_id for h in hits]
        if not hits:
            return self._response(question, ABSTAIN_MESSAGE, "abstained", reasons=["no_relevant_sources"])

        with tracer.start_as_current_span("generate"):
            draft = self.answerer.answer(question, hits)
        if draft.blocked_reason:
            return self._response(question, BLOCKED_MESSAGE, "blocked", reasons=[draft.blocked_reason])
        if draft.abstained:
            return self._response(
                question, draft.text, "abstained", reasons=["answerer_abstained"], retrieved_ids=retrieved_ids
            )

        text, citations = self._validate_citations(draft.text, draft.cited_numbers, hits)
        if not citations:
            return self._response(
                question, ABSTAIN_MESSAGE, "abstained", reasons=["no_valid_citations"], retrieved_ids=retrieved_ids
            )

        with tracer.start_as_current_span("safety.output"):
            verdict = self.safety.check_output(text)
        if not verdict.allowed:
            return self._response(question, BLOCKED_MESSAGE, "blocked", reasons=list(verdict.reasons))

        return self._response(
            question, text, "answered", citations=citations, degraded=draft.degraded, retrieved_ids=retrieved_ids
        )

    @staticmethod
    def _validate_citations(text: str, numbers: list[int], hits: list[SearchHit]) -> tuple[str, list[Citation]]:
        valid = [n for n in numbers if 1 <= n <= len(hits)]
        invalid = set(numbers) - set(valid)
        for n in invalid:  # remove markers that point at nothing rather than show a broken reference
            text = re.sub(rf"\s?\[{n}\]", "", text)
        return text.strip(), [Citation.from_chunk(n, hits[n - 1].chunk) for n in valid]


def build_agent(settings: Settings) -> PolicyAgent:
    """Pick Azure or offline providers per capability, based on what is configured."""
    from . import azure_clients
    from .corpus import load_corpus

    llm_client = azure_clients.openai_client(settings) if settings.azure_llm_enabled else None

    retriever: Retriever
    if settings.azure_search_enabled:
        embedder = Embedder(llm_client, settings.embedding_deployment, settings.embedding_dimensions)
        retriever = AzureSearchRetriever(
            azure_clients.search_client(settings),
            embedder,
            use_semantic_ranker=settings.use_semantic_ranker,
            min_reranker_score=settings.min_reranker_score,
        )
    else:
        retriever = BM25Retriever(load_corpus(settings.docs_dir))

    answerer: Answerer = (
        AzureOpenAIAnswerer(llm_client, settings.chat_deployment) if llm_client else ExtractiveAnswerer()
    )
    safety: SafetyScreen = (
        AzureContentSafetyScreen(azure_clients.content_safety_client(settings), settings.block_severity)
        if settings.azure_safety_enabled
        else RegexSafetyScreen()
    )
    return PolicyAgent(retriever, answerer, safety, top_k=settings.top_k, mode=settings.mode)
