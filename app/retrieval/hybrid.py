"""Hybrid dense+sparse retrieval with Reciprocal Rank Fusion (RRF) and listwise reranking.

Pipeline
--------
1. Embed the query with ``gemini-embedding-001`` (RETRIEVAL_QUERY task type).
2. Dense search: top-30 via Qdrant cosine ANN.
3. Sparse search: top-30 via BM25.
4. Fuse with RRF (k=60), keep top-20.
5. Reconstruct Chunk objects from the Qdrant payload store (scroll-based map).
6. Listwise rerank with Gemini 2.5 Flash, return top-5.

SOTA parameters (from atlas-rag-sota-v2):
  dense limit=30, BM25 limit=30, RRF k=60, post-fusion keep 20, final top_k=5.

Chunk reconstruction strategy
------------------------------
After RRF we have chunk_ids from both lists.  BM25-only ids may not appear in
the dense result payloads.  We scroll *all* Qdrant points once on first use to
build a ``chunk_id -> payload`` map; with <500 chunks this is a single call.

Module-level lazy initialisation
----------------------------------
``retrieve()`` is a convenience wrapper that constructs a ``HybridRetriever``
pointed at ``./qdrant_data`` and ``./bm25.json`` on first call and caches it in
the module global ``_default_retriever``.  Import the class directly for tests
or when custom paths are needed.
"""

from __future__ import annotations

import logging
import threading

from app.retrieval.bm25 import BM25Index
from app.retrieval.chunker import Chunk
from app.retrieval.embedder import embed
from app.retrieval.qdrant_store import Store
from app.retrieval.reranker import listwise_rerank

logger = logging.getLogger(__name__)

# Default on-disk paths (relative to repo root; override in tests)
_DEFAULT_QDRANT_PATH = "./qdrant_data"
_DEFAULT_BM25_PATH = "./bm25.json"

# ---------------------------------------------------------------------------
# Pure RRF helper (offline-testable, no I/O)
# ---------------------------------------------------------------------------


def rrf(
    dense_ids: list[str],
    sparse_ids: list[str],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_ids:
        Ordered list of chunk_ids from dense search (rank 0 = best).
    sparse_ids:
        Ordered list of chunk_ids from BM25 search (rank 0 = best).
    k:
        RRF smoothing constant.  Canonical value is 60 (per SOTA-RAG research).

    Returns
    -------
    List of ``(chunk_id, score)`` tuples sorted descending by score.
    Ties are broken by chunk_id (lexicographic ascending) for determinism.
    """
    scores: dict[str, float] = {}

    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    # Sort descending by score; stable tie-break on chunk_id ascending
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# Chunk reconstruction helper
# ---------------------------------------------------------------------------


def _payload_to_chunk(payload: dict) -> Chunk:
    """Reconstruct a :class:`Chunk` from a Qdrant point payload dict."""
    # Canonical fields stored explicitly by point_from_chunk
    doc_id = payload.get("doc_id", "")
    chunk_id = payload.get("chunk_id", "")
    text = payload.get("text", "")
    page = payload.get("page")

    # Everything else is doc-level metadata
    _canonical = {"doc_id", "chunk_id", "text", "page"}
    metadata = {k: v for k, v in payload.items() if k not in _canonical}

    return Chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        page=page,
        char_start=0,
        char_end=len(text),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# HybridRetriever class
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combines dense (Qdrant) + sparse (BM25) retrieval with RRF + Flash reranking.

    Parameters
    ----------
    store:
        Initialised :class:`~app.retrieval.qdrant_store.Store`.
    bm25:
        Loaded :class:`~app.retrieval.bm25.BM25Index`.
    """

    def __init__(self, store: Store, bm25: BM25Index) -> None:
        self.store = store
        self.bm25 = bm25
        self._payload_map: dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_payload_map(self) -> dict[str, dict]:
        """Scroll all Qdrant points once and build chunk_id -> payload map."""
        if self._payload_map is not None:
            return self._payload_map

        payload_map: dict[str, dict] = {}
        offset = None

        while True:
            records, next_offset = self.store.client.scroll(
                collection_name=self.store.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload:
                    cid = record.payload.get("chunk_id")
                    if cid:
                        payload_map[cid] = record.payload

            if next_offset is None:
                break
            offset = next_offset

        self._payload_map = payload_map
        return payload_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        rerank: bool = True,
    ) -> list[Chunk]:
        """Run the full hybrid pipeline for *query*.

        Parameters
        ----------
        query:
            Natural-language retrieval query.
        top_k:
            Number of final chunks to return.
        rerank:
            If True (default), apply Gemini Flash listwise reranking after RRF.
            Set to False to skip the reranker and return the top-K by RRF score —
            faster by ~8-15s per call, at a small relevance quality cost.

        Returns
        -------
        Up to *top_k* :class:`~app.retrieval.chunker.Chunk` objects.
        """
        # 1. Embed query
        qv = embed([query], "RETRIEVAL_QUERY")[0]

        # 2. Dense search (top-30)
        dense_hits = self.store.search(qv, limit=30)
        dense_ids = [p.payload["chunk_id"] for p in dense_hits if p.payload]

        # 3. Sparse search (top-30)
        sparse_hits = self.bm25.search(query, limit=30)
        sparse_ids = [cid for cid, _score in sparse_hits]

        # 4. RRF fusion, keep top-20
        fused = rrf(dense_ids, sparse_ids, k=60)[:20]

        if not fused:
            return []

        # 5. Reconstruct Chunk objects
        payload_map = self._ensure_payload_map()
        chunks: list[Chunk] = []
        missing = 0
        for cid, _score in fused:
            payload = payload_map.get(cid)
            if payload:
                chunks.append(_payload_to_chunk(payload))
            else:
                missing += 1

        total = len(fused)
        if missing:
            logger.warning(
                "retrieve: %d/%d fused chunk_ids missing from payload map "
                "(BM25/Qdrant snapshot mismatch?)",
                missing,
                total,
            )

        if not chunks:
            return []

        # 6. Optionally rerank with Gemini Flash, take top_k
        if rerank:
            return listwise_rerank(query, chunks, top_k=top_k)
        else:
            return chunks[:top_k]


# ---------------------------------------------------------------------------
# Module-level lazy convenience wrapper
# ---------------------------------------------------------------------------

_default_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def retrieve(query: str, top_k: int = 5, *, rerank: bool = True) -> list[Chunk]:
    """Module-level convenience wrapper.

    Lazily constructs a :class:`HybridRetriever` pointed at ``./qdrant_data``
    and ``./bm25.json`` on first call and caches it for subsequent calls.

    Thread-safe: a double-checked lock prevents duplicate construction when
    FastAPI's threadpool hits this path concurrently on first request.

    Parameters
    ----------
    query:
        Natural-language retrieval query.
    top_k:
        Number of chunks to return (default 5).
    rerank:
        Apply Gemini Flash listwise reranking (default True).  Pass False to
        skip reranking and return the top-K by RRF score — saves ~8-15s per
        call when throughput matters more than marginal relevance gain.

    Returns
    -------
    List of :class:`~app.retrieval.chunker.Chunk` objects.
    """
    global _default_retriever

    if _default_retriever is None:
        with _retriever_lock:
            if _default_retriever is None:
                store = Store(path=_DEFAULT_QDRANT_PATH)
                bm25 = BM25Index.load(_DEFAULT_BM25_PATH)
                _default_retriever = HybridRetriever(store=store, bm25=bm25)

    return _default_retriever.retrieve(query, top_k=top_k, rerank=rerank)
