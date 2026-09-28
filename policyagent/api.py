"""FastAPI surface: POST /ask and GET /health.

Run locally with `python -m policyagent serve` (or `uvicorn policyagent.api:app`).
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .agent import MAX_QUESTION_CHARS, PolicyAgent, build_agent
from .config import Settings
from .telemetry import setup_tracing


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


class CitationOut(BaseModel):
    number: int
    doc_id: str
    title: str
    section: str
    source: str


class AskResponse(BaseModel):
    answer: str
    outcome: str
    citations: list[CitationOut]
    reasons: list[str]
    degraded: bool


def create_app(agent: PolicyAgent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    exporter = setup_tracing(settings)
    agent = agent or build_agent(settings)
    app = FastAPI(title="Policy Q&A Agent", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": agent.mode, "tracing": exporter, "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        # Sync handler on purpose: the SDK clients are synchronous, and FastAPI runs
        # sync handlers in a thread pool so they do not block the event loop.
        result = agent.ask(request.question)
        return AskResponse(
            answer=result.answer,
            outcome=result.outcome,
            citations=[CitationOut(**c.__dict__) for c in result.citations],
            reasons=result.reasons,
            degraded=result.degraded,
        )

    return app


def __getattr__(name: str):  # lazily build `app` so importing this module has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
