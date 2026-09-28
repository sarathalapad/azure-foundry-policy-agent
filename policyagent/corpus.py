"""Load markdown policy documents and split them into section-aware chunks.

Policy documents are written in short headed sections, and a section is the
unit a citizen expects to be cited ("Retention periods"). Chunking on headings
first, and only then by size, keeps citations meaningful and avoids splitting
one rule across two chunks.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Return (metadata, body). Only flat `key: value` pairs are supported, which is all we need."""
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on sentence boundaries into windows of at most ~max_chars with a small overlap."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    windows, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            windows.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current = f"{current} {sentence}".strip()
    if current:
        windows.append(current.strip())
    return windows


def chunk_document(raw: str, source: str, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    doc_id = meta.get("doc_id", Path(source).stem)
    title = meta.get("title", Path(source).stem)

    # Pair each heading with the text that follows it. Text before the first
    # "##" (the title line and the fictional-document banner) is skipped: it is
    # boilerplate that would otherwise match every query.
    parts = _HEADING.split(body)
    chunks: list[Chunk] = []
    for i in range(1, len(parts), 3):
        level, heading, text = parts[i], parts[i + 1].strip(), parts[i + 2].strip()
        if level == "#" or not text:
            continue
        for piece in _split_long(" ".join(text.split()), max_chars, overlap):
            chunks.append(Chunk(f"{doc_id}-{len(chunks)}", doc_id, title, heading, piece, Path(source).name))
    return chunks


def load_corpus(docs_dir: str | Path) -> list[Chunk]:
    paths = sorted(Path(docs_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.name))
    return chunks
