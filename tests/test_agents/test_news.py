"""Live tests for the Stage-3e News specialist.

Makes real Vertex AI / Gemini Flash calls with Google Search grounding.
Marked 'live'; deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (pyproject.toml) means plain `async def` tests work.

Grounding is confirmed working on gg-gcpsbprojs-004 (~18-25 source URIs returned per
multi-theme query, validated items typically 3-6).  Assertions are therefore REAL gates:
  - test_has_grounded_items: >= 1 item required (grounding is enabled and ~28s call works).
  - test_items_have_real_uris: every item must have http URI, non-empty headline, non-empty tag.

If this test fails under Vertex quota pressure that is a real signal worth seeing.
The module-scoped fixture ensures only one grounding call per test run (~30-60s).
"""
import pytest

from app.agents import news
from app.schemas import NewsFindings

pytestmark = pytest.mark.live

_SUB_QUESTIONS = [
    "What are the latest Gulf real-estate market developments relevant to GCC investors?",
    "What are recent US tech-sector valuation or earnings stories this week?",
    "Are there any new green-bond issuances or ESG finance stories relevant to EU investors?",
    "What is the latest Nordic macro news (Riksbank, Swedish/Norwegian economy)?",
]


@pytest.fixture(scope="module")
async def news_findings():
    """Call the news specialist once; reuse across all tests in this module.

    Prints grounding reality so it is visible in CI / human review.
    """
    # Monkey-patch _pass1_grounded_fetch to capture grounding URI count
    import app.agents.news as _news_module

    original_pass1 = _news_module._pass1_grounded_fetch
    _captured: dict = {"uri_count": None}

    def _instrumented_pass1(questions):
        grounded_text, real_uris = original_pass1(questions)
        _captured["uri_count"] = len(real_uris)
        return grounded_text, real_uris

    _news_module._pass1_grounded_fetch = _instrumented_pass1

    try:
        result = await news.run(_SUB_QUESTIONS)
    finally:
        _news_module._pass1_grounded_fetch = original_pass1

    uri_count = _captured["uri_count"]

    print("\n--- News specialist grounding reality ---")
    if uri_count is None:
        print("  Pass-1 grounding: EXCEPTION (degraded to empty NewsFindings)")
    elif uri_count == 0:
        print("  Pass-1 grounding: RETURNED 0 URIs (grounding likely unavailable for this project)")
        print("  -> Degraded gracefully to NewsFindings(items=[])")
    else:
        print(f"  Pass-1 grounding: RETURNED {uri_count} URI(s) [OK]")

    print(f"\n  Validated items: {len(result.items)}")
    for item in result.items:
        safe_headline = item.headline.encode("ascii", errors="replace").decode("ascii")
        print(
            f"    headline={safe_headline[:90]!r}\n"
            f"      source_uri={item.source_uri!r}\n"
            f"      relevance_tag={item.relevance_tag!r}\n"
            f"      ts={item.ts.isoformat()!r}"
        )

    return result


async def test_returns_newsfindings(news_findings):
    """The news specialist must return a NewsFindings instance (always true, even if empty)."""
    assert isinstance(news_findings, NewsFindings), (
        f"Expected NewsFindings, got {type(news_findings)!r}"
    )


async def test_has_grounded_items(news_findings):
    """Grounding is confirmed working on gg-gcpsbprojs-004; at least 1 real item expected.

    If this fails it means grounding genuinely failed (quota, API change, etc.) —
    a real signal worth surfacing rather than hiding behind a lenient pass.
    """
    assert len(news_findings.items) >= 1, (
        "News specialist returned 0 validated items. "
        "Google Search grounding appears to have failed — check Vertex quota / logs."
    )


async def test_items_have_real_uris(news_findings):
    """Every validated item must have an http URI, non-empty headline, and non-empty tag."""
    for item in news_findings.items:
        assert item.source_uri.startswith("http"), (
            f"source_uri does not start with 'http': {item.source_uri!r} "
            f"(headline={item.headline!r})"
        )
        assert item.headline and item.headline.strip(), (
            f"headline is empty for item with source_uri={item.source_uri!r}"
        )
        assert item.relevance_tag and item.relevance_tag.strip(), (
            f"relevance_tag is empty for item with source_uri={item.source_uri!r}"
        )
