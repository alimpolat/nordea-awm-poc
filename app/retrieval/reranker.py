"""Gemini 2.5 Flash listwise relevance reranker.

Given a query and a list of candidate Chunks, sends a single structured-output
call to Gemini Flash asking it to rank the passages by relevance.  The model
returns a JSON array of 0-based passage indices (most relevant first).

Design notes
------------
* ``generate`` is imported at module level so tests can monkeypatch it cleanly.
* Passages are numbered 0-based in the prompt; the returned indices map directly
  to Python list positions.
* Robustness: out-of-range indices are silently dropped; any index the model
  omits is appended to the end in original order so no candidate is ever lost.
* If ``response.parsed`` is None (blocked / malformed response) the function
  falls back to the original candidate order.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from app.llm.vertex_client import generate  # module-level import for monkeypatching
from app.retrieval.chunker import Chunk
from app.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

_PASSAGE_TRUNCATE = 500  # max chars per passage in the prompt


class Ranking(BaseModel):
    """Structured output from the listwise reranker."""

    ranking: list[int]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a relevance ranker. "
    "Given a query and numbered passages, output a JSON array of the passage IDs "
    "in order of relevance, most relevant first."
)

_USER_TEMPLATE = (
    "Query: {query}\n"
    "Passages:\n"
    "{passages}\n"
    'Output: {{"ranking": [<ids in relevance order>]}}\n'
    "Passages are numbered starting at 0."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def listwise_rerank(
    query: str,
    candidates: list[Chunk],
    *,
    top_k: int | None = None,
    model: str | None = None,
) -> list[Chunk]:
    """Reorder *candidates* by Gemini Flash's judgment of relevance to *query*.

    Parameters
    ----------
    query:
        The retrieval query string.
    candidates:
        Candidate chunks to rerank (all are scored; order comes from RRF).
    top_k:
        If given, slice the reranked list to this length before returning.
    model:
        Override the Flash model name (defaults to ``settings.gemini_model_flash``).

    Returns
    -------
    Reordered list of Chunk objects.  Never shorter than *candidates* (before the
    optional ``top_k`` slice) — any index the model omits is appended in original
    order.
    """
    if not candidates:
        return []

    resolved_model = model or settings.gemini_model_flash

    # Build numbered passage block (0-based, truncated)
    passage_lines = []
    for i, chunk in enumerate(candidates):
        snippet = chunk.text[:_PASSAGE_TRUNCATE]
        passage_lines.append(f"[{i}] {snippet}")
    passages_text = "\n".join(passage_lines)

    user_text = _USER_TEMPLATE.format(query=query, passages=passages_text)

    try:
        response = generate(
            model=resolved_model,
            contents=user_text,
            system_instruction=_SYSTEM,
            response_schema=Ranking,
        )
    except Exception:
        logger.exception(
            "listwise_rerank: generate() failed, falling back to original candidate order"
        )
        result = list(candidates)
        return result[:top_k] if top_k is not None else result

    # Guard: parsed can be None on blocked / malformed responses
    if response.parsed is None:
        logger.warning(
            "listwise_rerank: response.parsed is None, falling back to original order"
        )
        result = list(candidates)
        return result[:top_k] if top_k is not None else result

    raw_ranking: list[int] = response.parsed.ranking
    n = len(candidates)

    # Normalise: keep only valid, in-range, first-occurrence indices
    seen: set[int] = set()
    ordered_indices: list[int] = []
    for idx in raw_ranking:
        if isinstance(idx, int) and 0 <= idx < n and idx not in seen:
            ordered_indices.append(idx)
            seen.add(idx)

    # Append any omitted candidates in original order
    for i in range(n):
        if i not in seen:
            ordered_indices.append(i)

    result = [candidates[i] for i in ordered_indices]
    return result[:top_k] if top_k is not None else result
