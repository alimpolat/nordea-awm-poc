"""Stage 3c — Macro Research specialist.

Receives sub-questions from the Planner, runs hybrid RAG retrieval for each
question, then asks Gemini Flash to synthesise MacroFindings citing ONLY the
retrieved chunks.

Design decisions
----------------
* For each sub-question we retrieve top-5 chunks via the full hybrid pipeline
  (embed → Qdrant dense → BM25 sparse → RRF → Flash rerank).  All per-question
  results are merged and deduped by chunk_id, capped at 12 chunks, to bound the
  prompt size.
* The model is instructed (in macro.txt) to cite only provided chunk_ids.  After
  the call we validate each EvidenceRef: any chunk_id not in the retrieved set
  is silently dropped, preventing hallucinated citations.  A finding with zero
  evidence_chunks after validation is kept (the synthesizer can downweight it);
  the claim itself is not discarded.
* Chunk text is truncated to ~600 characters to control token usage.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents._base import run_agent_sync
from app.retrieval.chunker import Chunk
from app.retrieval.hybrid import retrieve
from app.schemas import EvidenceRef, MacroFinding, MacroFindings

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_TOP_K_PER_QUESTION = 5
_MAX_TOTAL_CHUNKS = 12
_CHUNK_TEXT_MAX_CHARS = 600
# Retrieve for ALL planner sub-questions (no artificial cap).
# The ThreadPoolExecutor runs all retrieve() calls concurrently, so wall-clock cost
# is O(single retrieval latency) regardless of question count.  Brief generation
# runs in the background (cache-first serving), so latency is not on the user's
# critical path — fidelity and recall are the priority.

_DEFAULT_QUESTIONS: list[str] = [
    "What are the key macro developments this week that affect the Bergström portfolio?",
]


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def run(sub_questions: list[str]) -> MacroFindings:
    """Async entry point — wraps the blocking pipeline in a thread pool."""
    return await asyncio.to_thread(_run_sync, sub_questions)


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------


def _run_sync(sub_questions: list[str]) -> MacroFindings:
    questions = sub_questions if sub_questions else _DEFAULT_QUESTIONS[:]

    # 1. Retrieve chunks for ALL questions in parallel (ThreadPoolExecutor).
    # All questions are sent to both the retriever and the LLM for full coverage.
    all_chunks = _retrieve_and_merge(questions)

    # 2. Build the set of valid (doc_id, chunk_id) pairs for citation validation
    valid_pairs: set[tuple[str, str | None]] = {
        (c.doc_id, c.chunk_id) for c in all_chunks
    }

    # 3. Ask Flash to synthesise MacroFindings
    contents = _build_contents(questions, all_chunks)
    findings: MacroFindings = run_agent_sync("macro", contents, MacroFindings)

    # 4. Validate citations — drop hallucinated chunk_ids
    validated = _validate_citations(findings, valid_pairs)
    return validated


# ---------------------------------------------------------------------------
# Retrieval + dedup
# ---------------------------------------------------------------------------


def _retrieve_and_merge(questions: list[str]) -> list[Chunk]:
    """Retrieve top-K chunks per question in parallel, dedupe by chunk_id, cap at MAX_TOTAL_CHUNKS.

    Uses a ThreadPoolExecutor to run all per-question retrieve() calls concurrently.
    This reduces wall-clock time from O(N × retrieval_latency) to O(retrieval_latency)
    for N questions, which is critical for the orchestrator's parallel Stage-3 budget.
    """
    # Run all retrieve calls in parallel with the full reranker pipeline.
    # The listwise Flash reranker improves relevance quality of retrieved chunks,
    # which is the design-spec default.  Parallelism keeps wall-clock cost to
    # O(single retrieval latency) regardless of question count.
    per_question_results: dict[int, list[Chunk]] = {}
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = {
            pool.submit(retrieve, q, _TOP_K_PER_QUESTION): i
            for i, q in enumerate(questions)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                per_question_results[idx] = future.result()
            except Exception:
                per_question_results[idx] = []

    # Merge in original question order (preserves priority), dedupe, cap
    seen_chunk_ids: set[str] = set()
    merged: list[Chunk] = []
    for i in sorted(per_question_results):
        for chunk in per_question_results[i]:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                merged.append(chunk)
            if len(merged) >= _MAX_TOTAL_CHUNKS:
                return merged

    return merged


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------


def _validate_citations(
    findings: MacroFindings,
    valid_pairs: set[tuple[str, str | None]],
) -> MacroFindings:
    """Drop any EvidenceRef whose chunk_id was not in the retrieved set.

    A MacroFinding whose evidence_chunks becomes empty after validation is
    KEPT (the synthesizer sees it and can downweight it).  We never discard
    findings — only the hallucinated citation entries within them.
    """
    validated_items: list[MacroFinding] = []

    # Build a flat set of retrieved chunk_ids for O(1) lookup
    valid_chunk_ids: set[str | None] = {cid for (_doc_id, cid) in valid_pairs}
    valid_doc_ids: set[str] = {doc_id for (doc_id, _cid) in valid_pairs}

    for finding in findings.items:
        clean_refs: list[EvidenceRef] = []
        for ref in finding.evidence_chunks:
            # Accept the ref if:
            #   (a) its chunk_id appears in the retrieved set, OR
            #   (b) it has no chunk_id but the doc_id is in the retrieved set
            #       (model cited the doc without a chunk_id — keep it, it's weakly grounded)
            if ref.chunk_id is not None and ref.chunk_id in valid_chunk_ids:
                clean_refs.append(ref)
            elif ref.chunk_id is None and ref.doc_id in valid_doc_ids:
                clean_refs.append(ref)
            # else: drop — hallucinated citation

        validated_items.append(MacroFinding(
            claim=finding.claim,
            evidence_chunks=clean_refs,
            confidence=finding.confidence,
            impact_on_portfolio=finding.impact_on_portfolio,
        ))

    return MacroFindings(items=validated_items)


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _build_contents(questions: list[str], chunks: list[Chunk]) -> str:
    """Build the user-turn string for the Macro LLM call.

    Sections:
      1. Sub-questions from the Planner.
      2. Retrieved chunks — doc_id, chunk_id, and truncated text.
      3. Task instruction.
    """
    # 1. Sub-questions
    q_lines = ["=== PLANNER SUB-QUESTIONS (macro specialist) ==="]
    for i, q in enumerate(questions, start=1):
        q_lines.append(f"  [{i}] {q}")
    questions_section = "\n".join(q_lines)

    # 2. Retrieved chunks
    chunk_lines = [
        f"=== RETRIEVED MACRO CORPUS CHUNKS ({len(chunks)} chunks) ===",
        "Cite ONLY these chunk_ids in evidence_chunks. Do not cite any other source.",
    ]
    for c in chunks:
        truncated_text = c.text[:_CHUNK_TEXT_MAX_CHARS]
        if len(c.text) > _CHUNK_TEXT_MAX_CHARS:
            truncated_text += " [...]"
        chunk_lines.append(
            f"\n  [CHUNK]"
            f"\n  doc_id:   {c.doc_id}"
            f"\n  chunk_id: {c.chunk_id}"
            f"\n  text: {truncated_text}"
        )
    chunks_section = "\n".join(chunk_lines)

    # 3. Task instruction
    task = (
        "TASK:\n"
        "Using ONLY the retrieved chunks above as evidence, produce MacroFindings "
        "that answer the Planner's sub-questions.\n\n"
        "Rules:\n"
        "  - Every claim must rest on the chunks. If chunks do not support a question, "
        "    respond with claim='Insufficient evidence to answer: <question>', "
        "    confidence=low_needs_verification, evidence_chunks=[].\n"
        "  - Do NOT cite doc_ids or chunk_ids that are not in the list above.\n"
        "  - cite the chunk_id exactly as shown above (copy-paste).\n"
        "  - impact_on_portfolio must reference the Bergström family's actual allocation "
        "    (Gulf real estate, US tech, Nordic equity, EU fixed income, alternatives).\n"
        "  - Produce one MacroFinding per sub-question (or fewer if sub-questions overlap)."
    )

    return "\n\n".join([questions_section, chunks_section, task])
