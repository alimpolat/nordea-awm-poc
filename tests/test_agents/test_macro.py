"""Live tests for the Stage-3c Macro Research specialist.

Makes real Vertex AI / Gemini Flash calls (embed + rerank + synthesis).
Marked 'live'; deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (pyproject.toml) means plain `async def` tests work.

Shape-based asserts — robust to which claims the model synthesises.
"""
import pytest

from app.agents import macro

pytestmark = pytest.mark.live

_SUB_QUESTIONS = [
    "What is the current ECB rate path and how does it affect EU fixed-income duration risk?",
    "What does the IMF or BIS say about GCC / Gulf economic outlook and oil-price sensitivity?",
    "Are there any BIS systemic risk flags relevant to the Bergström portfolio's alternatives sleeve?",
]

# Known doc_id prefixes from the macro corpus
_KNOWN_DOC_PREFIXES = ("bis_", "ecb_", "imf_")


@pytest.fixture(scope="module")
async def macro_findings():
    """Call the agent once; reuse across all tests in this module."""
    result = await macro.run(_SUB_QUESTIONS)

    # Print for CI visibility and human eyeball check
    print("\n--- Macro Research findings ---")
    for finding in result.items:
        cited_doc_ids = [ref.doc_id for ref in finding.evidence_chunks]
        safe_claim = finding.claim.encode("ascii", errors="replace").decode("ascii")
        print(
            f"  claim={safe_claim[:120]!r}  "
            f"confidence={finding.confidence!r}  "
            f"cited_doc_ids={cited_doc_ids}"
        )
    return result


async def test_returns_at_least_one_finding(macro_findings):
    """The macro specialist must return at least one MacroFinding."""
    assert len(macro_findings.items) >= 1, (
        f"Expected ≥1 macro findings, got {len(macro_findings.items)}"
    )


async def test_at_least_one_finding_has_evidence(macro_findings):
    """At least one finding must carry non-empty evidence_chunks.

    (If ALL findings have zero evidence, the corpus retrieval pipeline
    may be broken or the corpus is empty.)
    """
    has_evidence = any(
        len(f.evidence_chunks) > 0 for f in macro_findings.items
    )
    assert has_evidence, (
        "No macro finding has evidence_chunks. "
        "This suggests retrieval or citation validation failed. "
        f"Findings: {[(f.claim[:60], len(f.evidence_chunks)) for f in macro_findings.items]}"
    )


async def test_cited_doc_ids_are_non_empty_strings(macro_findings):
    """Every cited EvidenceRef must have a non-empty doc_id string."""
    for finding in macro_findings.items:
        for ref in finding.evidence_chunks:
            assert isinstance(ref.doc_id, str) and ref.doc_id.strip(), (
                f"EvidenceRef has empty doc_id in finding: {finding.claim[:80]!r}"
            )


async def test_confidence_values_valid(macro_findings):
    """Every finding must have a valid confidence value."""
    valid_confidences = {"high", "medium", "low_needs_verification"}
    for finding in macro_findings.items:
        assert finding.confidence in valid_confidences, (
            f"Invalid confidence={finding.confidence!r} in finding: {finding.claim[:80]!r}"
        )


async def test_at_least_one_known_corpus_doc_cited(macro_findings):
    """At least one cited doc_id should come from the macro corpus (bis_, ecb_, imf_)."""
    all_cited = [
        ref.doc_id
        for f in macro_findings.items
        for ref in f.evidence_chunks
    ]
    has_known_prefix = any(
        any(doc_id.startswith(prefix) for prefix in _KNOWN_DOC_PREFIXES)
        for doc_id in all_cited
    )
    # This is a soft sanity check — warn but don't hard-fail if the model only
    # emits low_needs_verification findings (possible if corpus is sparse).
    if not has_known_prefix:
        all_low = all(
            f.confidence == "low_needs_verification" for f in macro_findings.items
        )
        if not all_low:
            pytest.fail(
                f"No cited doc_id starts with {_KNOWN_DOC_PREFIXES}. "
                f"Cited: {all_cited}. "
                "Either the macro corpus is empty or citations were all stripped by validation."
            )
        # If all findings are low_needs_verification, no evidence is expected — pass.
