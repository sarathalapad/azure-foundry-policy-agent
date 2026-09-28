"""Tiny text utilities shared by the offline retriever and the extractive answerer."""

from __future__ import annotations

import re

_STOPWORD_TEXT = """a an and are as at be by can do does for from has have how i if in is it its long
many may me much my of on or our should so than that the their them then there these
they this to up was we what when where which who whom why will with within would you
your after before any must"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def _stem(token: str) -> str:
    """A deliberately light stemmer: enough to match 'records'/'record' and 'renewal'/'renew'."""
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "als", "al", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
