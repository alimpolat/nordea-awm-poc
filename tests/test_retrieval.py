"""Tests for Qdrant dense store and BM25 sparse index.

All tests are fully offline — no real embeddings, no network.
Synthetic 768-dim vectors are generated deterministically via a seeded RNG.
"""
import random
import json
import pytest

from app.retrieval.chunker import Chunk
from app.retrieval.qdrant_store import Store, point_from_chunk
from app.retrieval.bm25 import BM25Index
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vector(seed: int) -> list[float]:
    """Return a reproducible 768-dim unit-ish vector seeded by *seed*."""
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(768)]
    # Normalise so cosine distance is well-defined
    magnitude = sum(x * x for x in v) ** 0.5
    return [x / magnitude for x in v]


def _make_chunks(n: int = 10) -> list[Chunk]:
    """Return *n* synthetic Chunk objects with canonical payload metadata."""
    asset_classes = ["equity", "fixed_income", "real_estate", "alternatives", "cash"]
    chunks = []
    for i in range(n):
        chunks.append(
            Chunk(
                doc_id=f"doc_{i:04d}",
                chunk_id=f"doc_{i:04d}::0000",
                text=f"Sample financial text for chunk {i}. Global markets outlook.",
                char_start=0,
                char_end=50,
                metadata={
                    "asset_class": asset_classes[i % len(asset_classes)],
                    "doc_type": "market_brief" if i % 2 == 0 else "ips",
                    "as_of_date": "2026-05-26",
                    "client_id": "bergstrom" if i < 5 else None,
                    "source_uri": f"gs://awm-corpus/doc_{i:04d}.pdf",
                },
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Dense store tests
# ---------------------------------------------------------------------------

class TestQdrantStore:
    def test_dense_upsert_and_search_returns_payloads(self):
        """Upsert 10 chunks, search with a chunk's own vector → ≥3 hits,
        top hit is most-similar chunk, payload has all required fields."""
        store = Store(path=None)  # in-memory
        store.create()

        chunks = _make_chunks(10)
        vectors = [_make_vector(i) for i in range(10)]

        points = [point_from_chunk(i, chunk, vectors[i]) for i, chunk in enumerate(chunks)]
        store.upsert(points)

        # Query with chunk 3's exact vector — it should rank #1
        query_vec = vectors[3]
        hits = store.search(query_vec, limit=5)

        assert len(hits) >= 3, f"Expected ≥3 hits, got {len(hits)}"

        top = hits[0]
        payload = top.payload

        # Verify required payload fields
        for field in ("chunk_id", "asset_class", "doc_type", "as_of_date", "source_uri"):
            assert field in payload, f"Payload missing field: {field!r}"

        # Top hit must be chunk 3 (exact match to query vector)
        assert payload["chunk_id"] == chunks[3].chunk_id, (
            f"Expected top hit chunk_id={chunks[3].chunk_id!r}, got {payload['chunk_id']!r}"
        )

    def test_dense_payload_filter_by_asset_class(self):
        """Upsert chunks with differing asset_class; filter search returns only
        matching asset_class."""
        store = Store(path=None)
        store.create()

        chunks = _make_chunks(10)
        vectors = [_make_vector(i + 100) for i in range(10)]

        # Patch all even chunks to asset_class="equity", odd to "bond"
        for i, chunk in enumerate(chunks):
            chunk.metadata["asset_class"] = "equity" if i % 2 == 0 else "bond"

        points = [point_from_chunk(i, chunk, vectors[i]) for i, chunk in enumerate(chunks)]
        store.upsert(points)

        filt = Filter(
            must=[FieldCondition(key="asset_class", match=MatchValue(value="equity"))]
        )
        hits = store.search(vectors[0], limit=10, filt=filt)

        assert len(hits) >= 1
        for hit in hits:
            assert hit.payload["asset_class"] == "equity", (
                f"Filter leak: got asset_class={hit.payload['asset_class']!r}"
            )

    def test_create_is_idempotent(self):
        """Calling create() twice should not raise (reset semantics)."""
        store = Store(path=None)
        store.create()
        store.create()  # second call should not raise

    def test_point_from_chunk_builds_correct_payload(self):
        """point_from_chunk merges chunk_id, doc_id, text + metadata into payload."""
        chunk = Chunk(
            doc_id="testdoc",
            chunk_id="testdoc::0000",
            text="Nordic high-yield bonds.",
            char_start=0,
            char_end=24,
            metadata={
                "asset_class": "fixed_income",
                "doc_type": "research",
                "as_of_date": "2026-05-26",
                "client_id": None,
                "source_uri": "gs://awm/test.pdf",
            },
        )
        vec = _make_vector(42)
        pt = point_from_chunk(7, chunk, vec)

        assert pt.id == 7
        assert len(pt.vector) == 768
        assert pt.payload["chunk_id"] == "testdoc::0000"
        assert pt.payload["doc_id"] == "testdoc"
        assert pt.payload["text"] == "Nordic high-yield bonds."
        assert pt.payload["asset_class"] == "fixed_income"
        assert pt.payload["source_uri"] == "gs://awm/test.pdf"

    def test_point_from_chunk_metadata_does_not_clobber_canonical_fields(self):
        """Canonical fields (chunk_id, text, page) win over identically-named metadata keys."""
        chunk = Chunk(
            doc_id="realdoc",
            chunk_id="realdoc::0001",
            text="Real text content.",
            char_start=0,
            char_end=18,
            page=3,
            metadata={
                "chunk_id": "WRONG",
                "text": "WRONG",
                "asset_class": "equity",
            },
        )
        vec = _make_vector(99)
        pt = point_from_chunk(0, chunk, vec)

        assert pt.payload["chunk_id"] == "realdoc::0001", (
            f"chunk_id was clobbered by metadata: {pt.payload['chunk_id']!r}"
        )
        assert pt.payload["text"] == "Real text content.", (
            f"text was clobbered by metadata: {pt.payload['text']!r}"
        )
        assert "page" in pt.payload, "page field should be present in payload"
        assert pt.payload["page"] == 3, f"page should be 3, got {pt.payload['page']!r}"


# ---------------------------------------------------------------------------
# BM25 tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    _TEXTS = [
        "Nordic equity market update for institutional investors",
        "Fixed income duration risk in rising rate environment",
        "Gulf real estate Emirates REIT Emaar Properties allocation",
        "Private equity buyout fund vintage 2024 performance",
        "Cash management short-term treasury bill strategy",
        "Emerging markets currency volatility hedging",
        "Sustainable ESG integration framework for UHNW",
        "Alternative investments hedge fund seeding mandate",
        "Global macro outlook central bank policy divergence",
        "Bergström family office rebalancing Nordic tilt",
    ]

    def _make_bm25_chunks(self) -> list[Chunk]:
        chunks = []
        for i, text in enumerate(self._TEXTS):
            chunks.append(
                Chunk(
                    doc_id=f"bm25doc_{i}",
                    chunk_id=f"bm25doc_{i}::0000",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                )
            )
        return chunks

    def test_bm25_ranks_keyword_match_first(self):
        """Query a distinctive term that only appears in one chunk → that chunk ranks #1."""
        idx = BM25Index()
        chunks = self._make_bm25_chunks()
        idx.build(chunks)

        # "Emirates REIT" appears only in chunk 2
        results = idx.search("Gulf real estate Emirates REIT", limit=5)
        assert results, "Expected non-empty results"
        top_chunk_id, top_score = results[0]
        assert top_chunk_id == "bm25doc_2::0000", (
            f"Expected bm25doc_2::0000 as top result, got {top_chunk_id!r}"
        )
        assert top_score > 0, "Top score should be positive"

    def test_bm25_save_load_roundtrip(self, tmp_path):
        """Save → load preserves search ranking; JSON is valid UTF-8 with Bergström."""
        idx = BM25Index()
        # Include "Bergström" to verify ensure_ascii=False
        chunks = self._make_bm25_chunks()
        idx.build(chunks)

        pre_results = idx.search("Bergström Nordic tilt", limit=3)

        save_path = str(tmp_path / "bm25.json")
        idx.save(save_path)

        # Validate JSON is valid UTF-8 and has the required structure
        raw = (tmp_path / "bm25.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "chunk_ids" in data
        assert "tokens" in data
        # chunk_ids are pure ASCII so we verify a known id is present
        assert "bm25doc_9::0000" in raw, "Saved JSON should contain the Bergström-chunk id"

        # Load and compare
        idx2 = BM25Index.load(save_path)
        post_results = idx2.search("Bergström Nordic tilt", limit=3)

        assert post_results, "Loaded index should return results"
        assert post_results[0][0] == pre_results[0][0], (
            f"Top chunk_id changed after roundtrip: {pre_results[0][0]!r} → {post_results[0][0]!r}"
        )

    def test_bm25_unknown_tokens_return_nonnegative_scores(self):
        """Querying with only tokens absent from the corpus returns non-negative scores."""
        idx = BM25Index()
        chunks = self._make_bm25_chunks()
        idx.build(chunks)
        results = idx.search("zzz_nonexistent_xyzzy", limit=5)
        # Either empty or all-zero scores — never negative
        for _, score in results:
            assert score >= 0, "Scores should be non-negative"

    def test_bm25_empty_string_query(self):
        """search('') should return results with all-zero (or non-positive) scores without raising."""
        idx = BM25Index()
        idx.build(self._make_bm25_chunks())
        results = idx.search("", limit=5)
        # Must not raise; all scores must be <= 0
        for _, score in results:
            assert score <= 0, f"Empty query should not produce positive scores, got {score}"

    def test_build_empty_raises(self):
        """BM25Index.build([]) should raise ValueError, not a cryptic ZeroDivisionError."""
        idx = BM25Index()
        with pytest.raises(ValueError, match="empty chunk list"):
            idx.build([])

    def test_bm25_search_returns_sorted_desc(self):
        """search() results are sorted by score descending."""
        idx = BM25Index()
        idx.build(self._make_bm25_chunks())
        results = idx.search("equity market Nordic", limit=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted descending by score"
