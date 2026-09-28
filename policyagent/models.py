"""Plain data types shared by every provider.

Keeping these free of any SDK types means the agent, the API and the evaluator
never care whether a result came from Azure or from the offline implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Outcome = Literal["answered", "abstained", "blocked"]


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of a source document."""

    chunk_id: str  # "<doc_id>-<n>", safe as an Azure AI Search key
    doc_id: str
    title: str
    section: str
    text: str
    source: str  # file name, shown to users as the citation target


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    number: int  # the [n] marker used in the answer text
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source: str

    @classmethod
    def from_chunk(cls, number: int, chunk: Chunk) -> Citation:
        return cls(number, chunk.chunk_id, chunk.doc_id, chunk.title, chunk.section, chunk.source)


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def ok(cls) -> SafetyVerdict:
        return cls(True)

    @classmethod
    def block(cls, *reasons: str) -> SafetyVerdict:
        return cls(False, tuple(reasons))


@dataclass
class DraftAnswer:
    """What an answerer returns before the agent validates citations."""

    text: str
    cited_numbers: list[int] = field(default_factory=list)
    abstained: bool = False
    blocked_reason: str | None = None  # e.g. the model's own content filter fired
    degraded: bool = False  # True when a fallback answerer produced the text


@dataclass
class AgentResponse:
    question: str
    answer: str
    outcome: Outcome
    citations: list[Citation] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    mode: str = "offline"
    degraded: bool = False
    retrieved_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
