"""Tests for RRF fusion and Gemini Flash listwise reranker.

All tests are fully offline (no network, no real embeddings).
The ``generate`` function used by ``reranker.py`` is monkeypatched
with a lightweight stub that returns a pre-built Ranking object.

RRF mathematical verification (k=60, ranks are 0-based)
---------------------------------------------------------
dense  = ["a", "b", "c"]   → a at rank 0: 1/60, b at rank 1: 1/61, c at rank 2: 1/62
sparse = ["b", "a", "d"]   → b at rank 0: 1/60, a at rank 1: 1/61, d at rank 2: 1/62

Combined RRF scores:
  a = 1/60 + 1/61 ≈ 0.016667 + 0.016393 = 0.033060
  b = 1/61 + 1/60 ≈ 0.016393 + 0.016667 = 0.033060
  c = 1/62         ≈ 0.016129
  d = 1/62         ≈ 0.016129

a and b are tied at the top; the tie-break is lexicographic on chunk_id
("a" < "b"), so "a" ranks first, "b" second.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from app.retrieval.chunker import Chunk
from app.retrieval.hybrid import _payload_to_chunk, rrf
from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import Ranking, listwise_rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str = "sample text") -> Chunk:
    """Return a minimal synthetic Chunk."""
    return Chunk(
        doc_id="testdoc",
        chunk_id=chunk_id,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _stub_response(ranking: list[int]):
    """Return a stub response object whose .parsed is a Ranking with given indices."""
    return SimpleNamespace(parsed=Ranking(ranking=ranking))


def _stub_response_none_parsed():
    """Return a stub response object whose .parsed is None (simulates blocked output)."""
    return SimpleNamespace(parsed=None)


# ---------------------------------------------------------------------------
# RRF — pure function, no mocks needed
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_item_high_in_both_wins(self):
        """Items appearing in both lists should outscore items in only one list."""
        dense = ["a", "b", "c"]
        sparse = ["b", "a", "d"]
        result = rrf(dense, sparse, k=60)

        # Extract id->score mapping
        scores = {cid: score for cid, score in result}

        # a and b are tied; both must beat c and d
        assert scores["a"] > scores["c"]
        assert scores["b"] > scores["d"]

        # Verify exact scores (see module docstring for derivation; ranks are 0-based)
        expected_a = 1 / 60 + 1 / 61  # dense rank 0 + sparse rank 1
        expected_b = 1 / 61 + 1 / 60  # dense rank 1 + sparse rank 0
        expected_c = 1 / 62           # dense rank 2 only
        expected_d = 1 / 62           # sparse rank 2 only

        assert abs(scores["a"] - expected_a) < 1e-10, f"a score mismatch: {scores['a']}"
        assert abs(scores["b"] - expected_b) < 1e-10, f"b score mismatch: {scores['b']}"
        assert abs(scores["c"] - expected_c) < 1e-10, f"c score mismatch: {scores['c']}"
        assert abs(scores["d"] - expected_d) < 1e-10, f"d score mismatch: {scores['d']}"

        # Tie-break: "a" < "b" lexicographically → "a" must be first
        ids = [cid for cid, _ in result]
        assert ids[0] == "a", f"Expected 'a' first (lexicographic tie-break), got {ids[0]!r}"
        assert ids[1] == "b"

    def test_rrf_disjoint_lists(self):
        """Items from two non-overlapping lists each get one reciprocal-rank contribution."""
        dense = ["x", "y"]
        sparse = ["p", "q"]
        result = rrf(dense, sparse, k=60)

        scores = {cid: score for cid, score in result}
        assert set(scores.keys()) == {"x", "y", "p", "q"}

        # Top of each list scores highest in its list
        assert scores["x"] > scores["y"]   # 1/61 > 1/62
        assert scores["p"] > scores["q"]   # 1/61 > 1/62

        # Both list-toppers have equal scores (1/61 each); no single winner
        assert abs(scores["x"] - scores["p"]) < 1e-10

    def test_rrf_single_nonempty_list(self):
        """RRF with one empty list still works; output mirrors the non-empty list ordering."""
        dense = ["alpha", "beta", "gamma"]
        result = rrf(dense, [], k=60)

        ids = [cid for cid, _ in result]
        scores = {cid: score for cid, score in result}

        assert ids == ["alpha", "beta", "gamma"], "Order should follow dense list"
        assert scores["alpha"] > scores["beta"] > scores["gamma"]

    def test_rrf_empty_inputs_return_empty(self):
        """Both lists empty → empty output."""
        result = rrf([], [], k=60)
        assert result == []


# ---------------------------------------------------------------------------
# Listwise reranker — monkeypatched generate
# ---------------------------------------------------------------------------


class TestListwiseReranker:
    """All tests in this class monkeypatch ``reranker.generate`` to avoid Vertex calls."""

    def _chunks(self, n: int = 3) -> list[Chunk]:
        return [_make_chunk(f"c{i}", f"Text for chunk {i}") for i in range(n)]

    def test_listwise_rerank_reorders_by_model_ranking(self, monkeypatch):
        """Model returns [2, 0, 1] → output order is chunks[2], chunks[0], chunks[1]."""
        candidates = self._chunks(3)
        monkeypatch.setattr(reranker_module, "generate", lambda **kw: _stub_response([2, 0, 1]))

        result = listwise_rerank("query", candidates)

        assert len(result) == 3
        assert result[0].chunk_id == "c2"
        assert result[1].chunk_id == "c0"
        assert result[2].chunk_id == "c1"

    def test_listwise_rerank_partial_ranking_appends_missing(self, monkeypatch):
        """Model returns [1] only → output = [c1, c0, c2]; nothing dropped."""
        candidates = self._chunks(3)
        monkeypatch.setattr(reranker_module, "generate", lambda **kw: _stub_response([1]))

        result = listwise_rerank("query", candidates)

        assert len(result) == 3
        assert result[0].chunk_id == "c1"
        # c0 and c2 appended in original order
        assert result[1].chunk_id == "c0"
        assert result[2].chunk_id == "c2"

    def test_listwise_rerank_ignores_out_of_range_indices(self, monkeypatch):
        """Model returns [9, 0] → 9 is out-of-range; output starts with c0, all 3 present."""
        candidates = self._chunks(3)
        monkeypatch.setattr(reranker_module, "generate", lambda **kw: _stub_response([9, 0]))

        result = listwise_rerank("query", candidates)

        assert len(result) == 3
        assert result[0].chunk_id == "c0"
        # c1 and c2 appended in original order
        assert result[1].chunk_id == "c1"
        assert result[2].chunk_id == "c2"

    def test_listwise_rerank_empty_candidates_returns_empty(self, monkeypatch):
        """Empty candidate list → empty output without calling generate."""
        called = []
        monkeypatch.setattr(reranker_module, "generate", lambda **kw: called.append(1))

        result = listwise_rerank("query", [])
        assert result == []
        assert called == [], "generate should not be called for empty candidates"

    def test_listwise_rerank_none_parsed_falls_back_to_original_order(self, monkeypatch):
        """response.parsed is None → return candidates in original order."""
        candidates = self._chunks(3)
        monkeypatch.setattr(
            reranker_module, "generate", lambda **kw: _stub_response_none_parsed()
        )

        result = listwise_rerank("query", candidates)

        assert len(result) == 3
        assert [c.chunk_id for c in result] == ["c0", "c1", "c2"]

    def test_listwise_rerank_none_parsed_respects_top_k(self, monkeypatch):
        """response.parsed is None with top_k=2 → first 2 candidates in original order."""
        candidates = self._chunks(3)
        monkeypatch.setattr(
            reranker_module, "generate", lambda **kw: _stub_response_none_parsed()
        )

        result = listwise_rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert [c.chunk_id for c in result] == ["c0", "c1"]

    def test_listwise_rerank_duplicate_indices_are_deduplicated(self, monkeypatch):
        """Model returns duplicate index [0, 0, 1] → c0 appears only once."""
        candidates = self._chunks(3)
        monkeypatch.setattr(
            reranker_module, "generate", lambda **kw: _stub_response([0, 0, 1])
        )

        result = listwise_rerank("query", candidates)

        assert len(result) == 3
        ids = [c.chunk_id for c in result]
        assert ids.count("c0") == 1, "Duplicate index should appear only once"
        assert ids[0] == "c0"
        assert ids[1] == "c1"

    def test_listwise_rerank_top_k_slices_result(self, monkeypatch):
        """top_k=2 with model returning [2, 0, 1] → only first 2 chunks returned."""
        candidates = self._chunks(3)
        monkeypatch.setattr(reranker_module, "generate", lambda **kw: _stub_response([2, 0, 1]))

        result = listwise_rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0].chunk_id == "c2"
        assert result[1].chunk_id == "c0"


# ---------------------------------------------------------------------------
# _payload_to_chunk — offline reconstruction helper
# ---------------------------------------------------------------------------


class TestPayloadToChunk:
    def test_payload_to_chunk_full(self):
        """A complete payload dict reconstructs a Chunk with core fields set and
        non-canonical keys landing in .metadata."""
        payload = {
            "chunk_id": "doc1_p3_c0",
            "doc_id": "doc1",
            "text": "Global equities rose 2% in Q1.",
            "page": 3,
            "asset_class": "equities",
            "doc_type": "ips",
            "as_of_date": "2026-01-01",
            "source_uri": "gs://bucket/doc1.pdf",
        }

        chunk = _payload_to_chunk(payload)

        assert chunk.chunk_id == "doc1_p3_c0"
        assert chunk.doc_id == "doc1"
        assert chunk.text == "Global equities rose 2% in Q1."
        assert chunk.page == 3
        assert chunk.char_start == 0
        assert chunk.char_end == len(payload["text"])

        # Non-canonical keys must appear in metadata
        for key in ("asset_class", "doc_type", "as_of_date", "source_uri"):
            assert key in chunk.metadata, f"Expected {key!r} in metadata"
            assert chunk.metadata[key] == payload[key]

        # Canonical keys must NOT bleed into metadata
        for key in ("chunk_id", "doc_id", "text", "page"):
            assert key not in chunk.metadata, f"{key!r} should not be in metadata"

    def test_payload_to_chunk_missing_keys_defaults(self):
        """A payload missing chunk_id and text yields a Chunk with empty-string
        defaults and does not raise — documents the current defensive behaviour."""
        payload: dict = {}  # completely empty

        chunk = _payload_to_chunk(payload)

        assert chunk.chunk_id == ""
        assert chunk.doc_id == ""
        assert chunk.text == ""
        assert chunk.page is None
        assert chunk.char_start == 0
        assert chunk.char_end == 0
        assert chunk.metadata == {}
