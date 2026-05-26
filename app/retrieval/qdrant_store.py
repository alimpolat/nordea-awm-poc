"""Qdrant embedded vector store for the AWM corpus.

Wraps ``qdrant_client.QdrantClient`` in a thin helper class that handles
collection (re)creation and dense-vector search.  Two modes:

* **In-memory** (tests):  ``Store(path=None)``
* **On-disk** (app):       ``Store(path="./qdrant_data")``

Collection name is ``awm_corpus`` with 768-dim cosine vectors, matching
``gemini-embedding-001`` output dimensions.

Point IDs are integer enumerate-indices (Qdrant requires unsigned-int or UUID).
The real ``chunk_id`` is stored in the point payload so that downstream code
can match results back to chunks.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.retrieval.chunker import Chunk

COLLECTION = "awm_corpus"
VECTOR_SIZE = 768


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def point_from_chunk(idx: int, chunk: Chunk, vector: list[float]) -> PointStruct:
    """Build a :class:`qdrant_client.models.PointStruct` from a *Chunk*.

    Parameters
    ----------
    idx:
        Integer point ID (enumerate index at upsert time).  Qdrant requires
        unsigned-int or UUID; arbitrary strings are not accepted.
    chunk:
        Source chunk.  ``chunk_id``, ``doc_id``, and ``text`` are merged into
        the payload alongside any doc-level fields from ``chunk.metadata``
        (``asset_class``, ``doc_type``, ``as_of_date``, ``client_id``,
        ``source_uri``).
    vector:
        Pre-computed 768-dim embedding vector.

    Returns
    -------
    :class:`PointStruct` ready for :py:meth:`Store.upsert`.
    """
    payload: dict = {
        **chunk.metadata,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "page": chunk.page,
    }
    return PointStruct(id=idx, vector=vector, payload=payload)


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------


class Store:
    """Thin wrapper around :class:`qdrant_client.QdrantClient`.

    Parameters
    ----------
    path:
        ``None``  → in-memory client (used in tests and CI).
        ``str``   → persistent on-disk client at the given directory path.

    Note
    ----
    Qdrant embedded mode writes a ``.lock`` file inside the data directory
    (e.g. ``./qdrant_data/.lock``).  Do **not** mount that directory as a
    strictly read-only volume.  Cloud Run's writable overlay layer is fine.
    """

    def __init__(self, path: str | None = None) -> None:
        self.client: QdrantClient = (
            QdrantClient(path=path) if path else QdrantClient(":memory:")
        )
        self.collection: str = COLLECTION

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create(self) -> None:
        """(Re)create the ``awm_corpus`` collection.

        Deletes the collection first if it already exists so that
        ``create()`` is safe to call multiple times (idempotent reset).
        Avoids the deprecated ``recreate_collection`` API.
        """
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: list[PointStruct]) -> None:
        """Upsert *points* into the collection.

        Parameters
        ----------
        points:
            List of :class:`PointStruct` objects, typically built via
            :func:`point_from_chunk`.
        """
        self.client.upsert(collection_name=self.collection, points=points)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 30,
        filt: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Dense-vector nearest-neighbour search.

        Parameters
        ----------
        vector:
            Query embedding (768-dim list of floats).
        limit:
            Maximum number of hits to return.
        filt:
            Optional :class:`qdrant_client.models.Filter` for metadata
            pre-filtering (e.g. by ``asset_class`` or ``client_id``).

        Returns
        -------
        List of :class:`qdrant_client.models.ScoredPoint` objects (sorted by
        score descending).  Each hit exposes ``.payload`` with the full
        chunk payload.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=filt,
            limit=limit,
            with_payload=True,
        )
        return response.points
