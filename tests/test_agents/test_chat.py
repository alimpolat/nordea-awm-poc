"""Live tests for the Task 4.4 ReAct chat agent.

Makes real Vertex AI / Gemini Flash calls with function calling (AFC or manual loop).
Marked 'live'; deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (pyproject.toml) means plain `async def` tests work.

Assertions are intentionally shape-first (non-deterministic ReAct path):
  - isinstance checks
  - non-empty answer
  - valid confidence enum value
  - non-empty cited_refs (chat has corpus + brief + holdings available)

The answers, refs, and confidence are printed so the build report can eyeball quality.
"""
import pytest

from app.agents import chat
from app.schemas import ChatRequest, ChatResponse

pytestmark = pytest.mark.live

_VALID_CONFIDENCE = {"high", "medium", "low_needs_verification"}


# ---------------------------------------------------------------------------
# Case 1: Gulf real estate exposure
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def gulf_response():
    """Ask about Gulf real estate exposure — exercises corpus + holdings tools."""
    req = ChatRequest(
        client_id="bergstrom",
        question="What is Bergström's Gulf real estate exposure and is it over target?",
    )
    resp = await chat.run(req)
    _print_response("Gulf real estate exposure", resp)
    return resp


async def test_gulf_response_is_chat_response(gulf_response):
    assert isinstance(gulf_response, ChatResponse), (
        f"Expected ChatResponse, got {type(gulf_response)!r}"
    )


async def test_gulf_response_answer_nonempty(gulf_response):
    assert gulf_response.answer and gulf_response.answer.strip(), (
        "ChatResponse.answer must be non-empty for Gulf real estate question"
    )


async def test_gulf_response_confidence_valid(gulf_response):
    assert gulf_response.confidence in _VALID_CONFIDENCE, (
        f"confidence must be one of {_VALID_CONFIDENCE!r}; got {gulf_response.confidence!r}"
    )


async def test_gulf_response_has_cited_refs(gulf_response):
    assert gulf_response.cited_refs, (
        "Expected at least one cited_ref for Gulf real estate question "
        "(portfolio or corpus should have been consulted)"
    )


# ---------------------------------------------------------------------------
# Case 2: Brent crude scenario
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def brent_response():
    """Ask a scenario question — exercises multiple tools (brief + corpus + possibly web)."""
    req = ChatRequest(
        client_id="bergstrom",
        question="What if Brent crude drops to $60 — how does that affect the portfolio?",
    )
    resp = await chat.run(req)
    _print_response("Brent crude $60 scenario", resp)
    return resp


async def test_brent_response_is_chat_response(brent_response):
    assert isinstance(brent_response, ChatResponse), (
        f"Expected ChatResponse, got {type(brent_response)!r}"
    )


async def test_brent_response_answer_nonempty(brent_response):
    assert brent_response.answer and brent_response.answer.strip(), (
        "ChatResponse.answer must be non-empty for Brent crude question"
    )


async def test_brent_response_confidence_valid(brent_response):
    assert brent_response.confidence in _VALID_CONFIDENCE, (
        f"confidence must be one of {_VALID_CONFIDENCE!r}; got {brent_response.confidence!r}"
    )


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def _print_response(label: str, resp: ChatResponse) -> None:
    """Print the response for the build report eyeball."""
    answer_preview = resp.answer[:400] + ("…" if len(resp.answer) > 400 else "")
    safe_answer = answer_preview.encode("ascii", errors="replace").decode("ascii")

    print(f"\n--- Chat ReAct: {label} ---")
    print(f"  confidence : {resp.confidence}")
    print(f"  answer     : {safe_answer}")
    print(f"  cited_refs ({len(resp.cited_refs)}):")
    for ref in resp.cited_refs:
        chunk_part = f"  chunk_id={ref.chunk_id!r}" if ref.chunk_id else ""
        uri_part = f"  uri={ref.source_uri!r}" if ref.source_uri else ""
        print(f"    doc_id={ref.doc_id!r}{chunk_part}{uri_part}")
