"""Offline contract tests for the sentence-aware character-window chunker.

No network required — the splitter is pure-Python and deterministic.
"""
import pytest

from app.retrieval.chunker import Chunk, chunk_document

# ~1300-char multi-sentence prose (simulates a single PDF page)
_SAMPLE_PAGE = (
    "Nordic equity markets posted a strong recovery last week, driven by robust earnings "
    "from the energy and shipping sectors. The Bergström family office holds a significant "
    "position in Nordic large-caps, currently running approximately five percentage points "
    "above the IPS target band of thirty-five percent. Currency headwinds from a stronger "
    "Swedish krona compressed USD-denominated returns by roughly two percent in the quarter. "
    "Gulf real estate allocations remain stable, with the Dubai residential sub-portfolio "
    "generating a net yield of four-point-three percent on a trailing twelve-month basis. "
    "The IPS mandates a maximum illiquid allocation of twenty percent; current illiquid "
    "exposure sits at eighteen percent, leaving a comfortable two-point buffer. Macro signals "
    "from the Federal Reserve suggest that the rate-cut cycle will extend through 2026, which "
    "broadly supports duration in the fixed-income sleeve. European investment-grade credit "
    "spreads have compressed to post-pandemic lows, reducing the relative attractiveness of "
    "the sleeve versus Nordic short-duration equivalents. Three next-best actions are proposed "
    "for Monday's review: trim US tech overweight, add Nordic short-duration bonds, and "
    "initiate a modest allocation to Gulf infrastructure funds."
)


def test_chunk_count_reasonable():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc")
    # A ~1300-char page with 300-700 char windows should yield 2-5 chunks
    assert 2 <= len(chunks) <= 5


def test_non_last_chunks_within_char_bounds():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc")
    for chunk in chunks[:-1]:
        assert 300 <= len(chunk.text) <= 700, (
            f"chunk {chunk.chunk_id} length {len(chunk.text)} out of [300, 700]"
        )


def test_last_chunk_not_empty():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc")
    assert len(chunks[-1].text) > 0


def test_consecutive_chunks_overlap():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc")
    if len(chunks) < 2:
        pytest.skip("need at least 2 chunks to test overlap")
    for i in range(len(chunks) - 1):
        tail = chunks[i].text[-60:]   # last 60 chars of chunk i
        head = chunks[i + 1].text[:200]  # first 200 chars of chunk i+1
        # At least some overlap text from tail should appear in head
        # We look for the longest common suffix/prefix overlap >= 30 chars
        overlap_found = any(tail[j:] in head for j in range(0, len(tail) - 29))
        assert overlap_found, (
            f"No overlap >= 30 chars between chunk {i} tail and chunk {i+1} head"
        )


def test_char_start_end_offsets_valid():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc")
    for chunk in chunks:
        assert chunk.char_start < chunk.char_end
        assert chunk.text == _SAMPLE_PAGE[chunk.char_start:chunk.char_end]


def test_chunk_ids_unique_and_stable():
    chunks_a = chunk_document(_SAMPLE_PAGE, doc_id="doc_abc")
    chunks_b = chunk_document(_SAMPLE_PAGE, doc_id="doc_abc")
    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    assert ids_a == ids_b  # deterministic
    assert len(ids_a) == len(set(ids_a))  # unique


def test_doc_id_propagated():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="bergstrom_ips")
    assert all(c.doc_id == "bergstrom_ips" for c in chunks)


def test_page_propagated():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc", page=3)
    assert all(c.page == 3 for c in chunks)


def test_metadata_passthrough():
    meta = {"doc_type": "ips", "asset_class": "equity", "source_uri": "gs://bucket/ips.pdf"}
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc", metadata=meta)
    assert all(c.metadata == meta for c in chunks)


def test_metadata_defaults_empty():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="test_doc")
    assert all(c.metadata == {} for c in chunks)


def test_chunk_model_fields():
    chunks = chunk_document(_SAMPLE_PAGE, doc_id="field_test")
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.doc_id == "field_test"
    assert isinstance(c.chunk_id, str)
    assert isinstance(c.text, str)
    assert c.page is None  # not passed
    assert isinstance(c.char_start, int)
    assert isinstance(c.char_end, int)
    assert isinstance(c.metadata, dict)


# ---------------------------------------------------------------------------
# Edge-case tests (offline, no markers)
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty_list():
    assert chunk_document("", doc_id="edge") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_document("   \n\t  ", doc_id="edge") == []


def test_single_short_sentence():
    text = "The fund returned twelve percent last year."
    chunks = chunk_document(text, doc_id="edge")
    assert len(chunks) == 1
    assert len(chunks[0].text) > 0
    assert chunks[0].text == text


def test_single_oversized_sentence_passthrough():
    # Build a sentence longer than _TARGET_MAX (700 chars) with no sentence-
    # terminal punctuation until the very end so it cannot be split further.
    long_sentence = "A" * 750 + "."
    chunks = chunk_document(long_sentence, doc_id="edge")
    assert len(chunks) == 1
    assert long_sentence in chunks[0].text
