"""Evaluation: offline checks that always run, plus Azure AI evaluators when configured.

The offline metrics are cheap and deterministic, so CI can fail on a regression
in citation validity or abstention behaviour. LLM-judged metrics (groundedness,
relevance) from `azure-ai-evaluation` add a quality signal but cost tokens and
vary between runs, so they are opt-in and run only with Azure credentials.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import PolicyAgent
from .config import REPO_ROOT, Settings
from .models import AgentResponse

log = logging.getLogger(__name__)
DEFAULT_DATASET = REPO_ROOT / "eval" / "golden.jsonl"


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected_doc: str | None
    keywords: list[str]
    expect_blocked: bool = False


def load_cases(path: Path = DEFAULT_DATASET) -> list[Case]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def score_case(case: Case, response: AgentResponse) -> dict[str, Any]:
    """Per-case metrics. None means 'not applicable' and is excluded from averages."""
    answered = response.outcome == "answered"
    retrieved = set(response.retrieved_ids)
    if case.expect_blocked:
        return {"id": case.id, "outcome": response.outcome, "safety_correct": response.outcome == "blocked"}
    should_answer = case.expected_doc is not None
    row: dict[str, Any] = {
        "id": case.id,
        "outcome": response.outcome,
        "abstention_correct": answered == should_answer,
        "citation_valid": None,
        "cited_expected_doc": None,
        "keyword_recall": None,
    }
    if answered:
        # A valid citation points at a chunk that was actually retrieved for this question.
        row["citation_valid"] = all(c.chunk_id in retrieved for c in response.citations) and bool(response.citations)
    if should_answer:
        row["cited_expected_doc"] = any(c.doc_id == case.expected_doc for c in response.citations)
        text = response.answer.lower() if answered else ""
        row["keyword_recall"] = sum(k.lower() in text for k in case.keywords) / len(case.keywords)
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = ["abstention_correct", "citation_valid", "cited_expected_doc", "keyword_recall", "safety_correct"]
    summary: dict[str, float] = {}
    for metric in metrics:
        values = [float(r[metric]) for r in rows if r.get(metric) is not None]
        if values:
            summary[metric] = round(sum(values) / len(values), 3)
    return summary


def azure_ai_scores(settings: Settings, pairs: list[tuple[str, AgentResponse, str]]) -> dict[str, float]:
    """Groundedness and relevance (1-5) via azure-ai-evaluation, averaged over answered cases."""
    from azure.ai.evaluation import AzureOpenAIModelConfiguration, GroundednessEvaluator, RelevanceEvaluator

    from .azure_clients import credential

    if not settings.openai_endpoint:
        log.warning("AZURE_OPENAI_ENDPOINT is required for the judge model; skipping Azure AI evaluators")
        return {}
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=settings.openai_endpoint,
        azure_deployment=settings.chat_deployment,
        api_version=settings.openai_api_version,
    )
    if settings.openai_api_key:
        model_config["api_key"] = settings.openai_api_key
    kwargs = {} if settings.openai_api_key else {"credential": credential()}
    groundedness = GroundednessEvaluator(model_config, **kwargs)
    relevance = RelevanceEvaluator(model_config, **kwargs)
    g, r = [], []
    for question, response, context in pairs:
        g.append(float(groundedness(query=question, response=response.answer, context=context)["groundedness"]))
        r.append(float(relevance(query=question, response=response.answer)["relevance"]))
    if not g:
        return {}
    return {"groundedness": round(sum(g) / len(g), 2), "relevance": round(sum(r) / len(r), 2)}


def run_evaluation(agent: PolicyAgent, settings: Settings, dataset: Path = DEFAULT_DATASET) -> dict[str, Any]:
    rows, judged = [], []
    for case in load_cases(dataset):
        response = agent.ask(case.question)
        rows.append(score_case(case, response))
        if response.outcome == "answered":
            hits = agent.retriever.search(case.question, agent.top_k)
            judged.append((case.question, response, "\n\n".join(h.chunk.text for h in hits)))
    report: dict[str, Any] = {"mode": agent.mode, "cases": len(rows), "offline": summarise(rows), "rows": rows}
    if settings.azure_llm_enabled:
        report["azure_ai_evaluation"] = azure_ai_scores(settings, judged)
    return report
