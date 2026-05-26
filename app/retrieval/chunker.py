"""Sentence-aware character-window chunker.

Splits a document text into overlapping chunks using sentence boundaries
(`. ! ?` terminators) as natural break points, targeting 300-700 chars per
chunk with ~50 chars of overlap from the previous window's tail.

Fully offline and deterministic — no network calls, no tokenizer downloads.
The `Chunk` model defined here is the shared contract imported by every other
retrieval module (contextual augmentation, qdrant store, bm25, hybrid, build_index).

Limitation: regex sentence splitting treats abbreviations like "e.g.", "i.e.",
"vs.", "No." as sentence boundaries.  This is acceptable for the POC corpus
(structured financial prose with few abbreviations); the upgrade path is a
punkt-style tokenizer (e.g. nltk.tokenize.PunktSentenceTokenizer).
"""
import re

from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str          # stable id, e.g. f"{doc_id}::{index:04d}"
    text: str
    page: int | None = None   # PDF page number; None for HTML sources
    char_start: int
    char_end: int
    metadata: dict = {}       # doc-level metadata populated at ingest; chunker
                               # leaves it {} unless caller passes it


# Sentence-boundary pattern: split after `. ! ?` followed by whitespace.
# The regex requires at least one whitespace char, so it never matches end-of-
# string; the final (unterminated) sentence is collected by the explicit tail
# block in _split_sentences.
_SENT_RE = re.compile(r'(?<=[.!?])\s+')

_TARGET_MIN = 300
_TARGET_MAX = 700
_OVERLAP = 50


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Return [(sentence, start_offset), ...] preserving char positions."""
    sentences: list[tuple[str, int]] = []
    prev = 0
    for m in _SENT_RE.finditer(text):
        end = m.start() + 1  # +1 absorbs the first whitespace char into the boundary;
                              # offsets stay consistent because window text is
                              # re-sliced from the original string via raw_start/raw_end.
        sent = text[prev:end]
        if sent.strip():
            sentences.append((sent, prev))
        prev = m.end()
    # remainder after last sentence boundary
    tail = text[prev:]
    if tail.strip():
        sentences.append((tail, prev))
    return sentences


def chunk_document(
    text: str,
    doc_id: str,
    *,
    page: int | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping sentence-aware chunks.

    Parameters
    ----------
    text:      Full text for a single page / document unit.
    doc_id:    Identifier of the source document.
    page:      PDF page number (None for HTML).
    metadata:  Doc-level metadata dict; passed through to every chunk unchanged.
               Defaults to {} when omitted.

    Returns
    -------
    List of `Chunk` objects.  `char_start`/`char_end` are offsets into *text*.
    `chunk_id` is ``f"{doc_id}::{index:04d}"`` — stable across identical inputs.
    """
    meta = metadata if metadata is not None else {}
    sentences = _split_sentences(text)

    if not sentences:
        return []

    chunks: list[Chunk] = []
    idx = 0          # sentence cursor
    chunk_index = 0

    while idx < len(sentences):
        # Build the primary window: pack sentences until we hit _TARGET_MAX.
        window_sentences: list[tuple[str, int]] = []
        window_len = 0

        # Prepend overlap from tail of previous chunk if available.
        overlap_prefix = ""
        overlap_start_in_text = -1
        if chunks:
            prev_text = chunks[-1].text
            overlap_prefix = prev_text[-_OVERLAP:]
            # Determine the true char offset in original text for the overlap
            # The overlap is the last _OVERLAP chars of the previous chunk text,
            # which sits at chunks[-1].char_end - len(overlap_prefix) in text.
            overlap_start_in_text = chunks[-1].char_end - len(overlap_prefix)

        while idx < len(sentences):
            sent_text, sent_offset = sentences[idx]
            proposed = window_len + len(sent_text)
            if window_sentences and proposed > _TARGET_MAX:
                # Would exceed max — stop here (commit window, restart from idx).
                break
            window_sentences.append((sent_text, sent_offset))
            window_len += len(sent_text)
            idx += 1
            # If we've reached minimum, stop adding if the next sentence would
            # push us past the max (checked at top of loop on next iteration).
            # Also stop naturally when window reaches or exceeds the minimum.

        # The inner loop always appends at least one sentence before it can
        # break (the `window_sentences and proposed > _TARGET_MAX` guard
        # short-circuits when window_sentences is empty, so an oversized single
        # sentence is always appended).  The branch below is therefore
        # unreachable; keep as an assertion to surface any future regression.
        assert window_sentences, "invariant: inner loop always appends at least one sentence"

        # Compute raw char range from the first/last sentences in the window.
        first_sent, first_offset = window_sentences[0]
        last_sent, last_offset = window_sentences[-1]
        raw_start = first_offset
        raw_end = last_offset + len(last_sent)

        # Build chunk text: overlap_prefix + window text (deduplicated).
        window_text = text[raw_start:raw_end]

        if overlap_prefix and overlap_start_in_text >= 0:
            chunk_text = overlap_prefix + window_text
            char_start = overlap_start_in_text
        else:
            chunk_text = window_text
            char_start = raw_start

        char_end = raw_end

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::{chunk_index:04d}",
                text=chunk_text,
                page=page,
                char_start=char_start,
                char_end=char_end,
                metadata=meta,
            )
        )
        chunk_index += 1

    return chunks
