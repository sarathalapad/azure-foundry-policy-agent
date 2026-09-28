from __future__ import annotations

from types import SimpleNamespace

import pytest

from policyagent.agent import PolicyAgent
from policyagent.answering import ExtractiveAnswerer
from policyagent.config import Settings
from policyagent.corpus import load_corpus
from policyagent.retrieval import BM25Retriever
from policyagent.safety import RegexSafetyScreen


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(Settings().docs_dir)


@pytest.fixture
def offline_agent(chunks):
    return PolicyAgent(BM25Retriever(chunks), ExtractiveAnswerer(), RegexSafetyScreen())


class FakeEmbeddings:
    """Stands in for client.embeddings: returns a fixed-size vector per input and records calls."""

    def __init__(self, dims: int = 8):
        self.dims, self.calls = dims, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [SimpleNamespace(index=i, embedding=[0.1 * (i + 1)] * self.dims) for i in range(len(kwargs["input"]))]
        return SimpleNamespace(data=list(reversed(data)))  # out of order on purpose


class FakeChat:
    def __init__(self, content: str | None = None, finish_reason: str = "stop", error: Exception | None = None):
        self.content, self.finish_reason, self.error, self.calls = content, finish_reason, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)])


def fake_openai_client(chat: FakeChat | None = None, embeddings: FakeEmbeddings | None = None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=chat or FakeChat("")), embeddings=embeddings or FakeEmbeddings()
    )
