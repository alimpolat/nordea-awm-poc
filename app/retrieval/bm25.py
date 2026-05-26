"""BM25 sparse retrieval index for the AWM corpus.

Uses ``rank_bm25.BM25Okapi`` backed by an in-process token store.  Designed
to complement the Qdrant dense index in the hybrid retrieval pipeline (Task 2.4).

Tokenisation
------------
Lowercase, split on non-alphanumeric: ``re.findall(r"[a-z0-9]+", text.lower())``.
This keeps tickers (``"reit"``, ``"esg"``), numbers, and bare words while
stripping punctuation.  Simple and deterministic — no external tokenizer.

Persistence
-----------
``save()`` serialises ``{chunk_ids, tokens}`` to a UTF-8 JSON file
(``ensure_ascii=False`` so Scandinavian names like *Bergström* are preserved).
``load()`` rebuilds the ``BM25Okapi`` from the saved token lists.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

from rank_bm25 import BM25Okapi

from app.retrieval.chunker import Chunk

# Tokeniser pattern: one or more lowercase letters or digits.
_TOK_RE: re.Pattern = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric."""
    return _TOK_RE.findall(text.lower())


class BM25Index:
    """In-process BM25 sparse index over :class:`~app.retrieval.chunker.Chunk` objects.

    Usage::

        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("Nordic equity allocation", limit=10)
        idx.save("./bm25.json")

        idx2 = BM25Index.load("./bm25.json")
    """

    _SAVE_ENCODING: ClassVar[str] = "utf-8"

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenise *chunks* and build the BM25 index in-process.

        Parameters
        ----------
        chunks:
            Source chunks (ordered; the parallel ``chunk_ids`` list mirrors
            this order so search results can be mapped back).
        """
        if not chunks:
            raise ValueError("Cannot build BM25Index from an empty chunk list.")
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 30) -> list[tuple[str, float]]:
        """Return top-*limit* ``(chunk_id, score)`` pairs sorted by score descending.

        Chunks with a BM25 score of zero are included in the output (BM25Okapi
        always returns a score for every document); callers can filter by score
        if needed.

        Parameters
        ----------
        query:
            Natural-language or keyword query string.
        limit:
            Maximum results to return.

        Returns
        -------
        List of ``(chunk_id, score)`` tuples, sorted by score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index has not been built — call build() or load() first.")

        q_tokens = _tokenize(query)
        scores: list[float] = self._bm25.get_scores(q_tokens).tolist()

        # Pair with chunk_ids and sort descending
        paired = sorted(
            zip(self._chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(paired[:limit])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the index to a UTF-8 JSON file at *path*.

        The saved JSON contains:

        * ``chunk_ids`` — ordered list of chunk id strings.
        * ``tokens``    — parallel list of token lists (so ``BM25Okapi`` can
          be reconstructed without re-reading the original corpus).

        ``ensure_ascii=False`` preserves Scandinavian characters (e.g.
        ``"Bergström"``) without escape sequences.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index has not been built — call build() first.")

        data = {
            "chunk_ids": self._chunk_ids,
            "tokens": self._tokens,
        }
        with open(path, "w", encoding=self._SAVE_ENCODING) as fh:
            json.dump(data, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Reconstruct a :class:`BM25Index` from a file written by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the JSON file produced by :meth:`save`.

        Returns
        -------
        A fully-initialised :class:`BM25Index` ready for :meth:`search`.
        """
        with open(path, encoding=cls._SAVE_ENCODING) as fh:
            data = json.load(fh)

        inst = cls()
        inst._chunk_ids = data["chunk_ids"]
        inst._tokens = data["tokens"]
        inst._bm25 = BM25Okapi(inst._tokens)
        return inst
