"""Offline contract tests for the Anthropic Contextual Retrieval augmentation module.

No network calls. The LLM (ctx.generate) is monkeypatched in every offline test.
"""
import json
from pathlib import Path

import pytest

import app.retrieval.contextual as ctx
from app.retrieval.chunker import Chunk


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DOC_ID = "bergstrom_q1"
_CHUNK_TEXT = (
    "Nordic equity markets posted a strong recovery last week, driven by robust "
    "earnings from the energy and shipping sectors."
)
_DOC_TEXT = (
    "Bergström Q1 Portfolio Statement — March 2026. "
    + _CHUNK_TEXT
    + " The Bergström family office holds a significant position in Nordic large-caps."
)
_CONTEXT_SENTENCE = "This chunk is from the Bergström Q1 portfolio statement."


def _make_chunk(chunk_id: str = "bergstrom_q1::0000") -> Chunk:
    return Chunk(
        doc_id=_DOC_ID,
        chunk_id=chunk_id,
        text=_CHUNK_TEXT,
        char_start=0,
        char_end=len(_CHUNK_TEXT),
    )


class _Resp:
    """Stub mimicking a google-genai response object."""

    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Offline tests (no live marker — these are the contract)
# ---------------------------------------------------------------------------


def test_augmented_text_has_context_prefix_then_chunk(monkeypatch, tmp_path):
    """contextualize() returns '<context>\n\n<original chunk text>'."""
    monkeypatch.setattr(ctx, "generate", lambda *a, **kw: _Resp(_CONTEXT_SENTENCE))

    chunk = _make_chunk()
    result = ctx.contextualize(chunk, _DOC_TEXT, cache_dir=tmp_path)

    expected = f"{_CONTEXT_SENTENCE}\n\n{_CHUNK_TEXT}"
    assert result == expected


def test_context_is_cached_no_second_llm_call(monkeypatch, tmp_path):
    """The LLM is called exactly once; the second call reads from the cache file."""
    call_count = {"n": 0}

    def _fake_generate(*args, **kwargs):
        call_count["n"] += 1
        return _Resp(_CONTEXT_SENTENCE)

    monkeypatch.setattr(ctx, "generate", _fake_generate)

    chunk = _make_chunk()
    ctx.contextualize(chunk, _DOC_TEXT, cache_dir=tmp_path)
    ctx.contextualize(chunk, _DOC_TEXT, cache_dir=tmp_path)

    assert call_count["n"] == 1, "LLM should be called exactly once; second hit must use cache"

    cache_file = tmp_path / f"{_DOC_ID}.json"
    assert cache_file.exists(), "Cache file must be created"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert chunk.chunk_id in data
    assert data[chunk.chunk_id] == _CONTEXT_SENTENCE


def test_none_text_response_falls_back_to_chunk_text(monkeypatch, tmp_path):
    """When response.text is None (safety block / thinking-only), contextualize()
    returns exactly chunk.text — no crash, no leading blank line — and the cache
    file either doesn't exist or does NOT contain an entry for that chunk_id
    (empty context must not be poisoned into the cache)."""

    class _NoneResp:
        text = None

    monkeypatch.setattr(ctx, "generate", lambda *a, **kw: _NoneResp())

    chunk = _make_chunk()
    result = ctx.contextualize(chunk, _DOC_TEXT, cache_dir=tmp_path)

    # Must fall back to bare chunk text — no crash, no leading blank line.
    assert result == _CHUNK_TEXT, (
        f"Expected exactly chunk.text on None response; got: {result!r}"
    )
    assert not result.startswith("\n"), "No leading newline when context is empty"

    # Cache must NOT contain an entry for this chunk_id.
    cache_file = tmp_path / f"{_DOC_ID}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert chunk.chunk_id not in data, (
            "Empty context must not be written to cache (would poison retries)"
        )


def test_corrupt_cache_file_is_tolerated(monkeypatch, tmp_path):
    """A corrupt (non-JSON) cache file must not raise — treat as empty and overwrite."""
    cache_file = tmp_path / f"{_DOC_ID}.json"
    cache_file.write_text("<<<NOT JSON>>>", encoding="utf-8")

    monkeypatch.setattr(ctx, "generate", lambda *a, **kw: _Resp(_CONTEXT_SENTENCE))

    chunk = _make_chunk()
    # Must not raise
    result = ctx.contextualize(chunk, _DOC_TEXT, cache_dir=tmp_path)

    expected = f"{_CONTEXT_SENTENCE}\n\n{_CHUNK_TEXT}"
    assert result == expected

    # Cache file must now contain valid JSON with the chunk_id entry
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data[chunk.chunk_id] == _CONTEXT_SENTENCE


# ---------------------------------------------------------------------------
# Live test (skipped unless -m live is passed)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_flash_context_is_non_empty_and_shorter_than_chunk(tmp_path):
    """Call real Flash; returned context is non-empty and at most 3 sentences (~600 chars)."""
    chunk = _make_chunk()
    context = ctx.context_for(chunk, _DOC_TEXT, cache_dir=tmp_path)

    assert context, "context_for() must return a non-empty string"
    # The prompt asks for 1-3 sentences. 600 chars is a generous upper bound for 3 sentences.
    assert len(context) <= 600, (
        f"Context ({len(context)} chars) exceeds 600-char upper bound — prompt may have been ignored"
    )
