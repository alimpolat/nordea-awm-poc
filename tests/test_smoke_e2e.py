"""End-to-end smoke test — runs the full 5-stage brief pipeline LIVE.

Marks: pytest.mark.live
Usage: uv run --no-sync pytest tests/test_smoke_e2e.py -q

The module-scoped fixture ``brief`` calls ``generate_brief`` ONCE so all
five assertions share a single pipeline run.  Wall-clock time is captured
in the fixture and asserted in test_completes_under_budget.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.orchestrator import generate_brief, save_brief_cache
from app.schemas import BriefSchema

# ---------------------------------------------------------------------------
# Mark every test in this module as "live"
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full pipeline ONCE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def brief_and_elapsed() -> tuple[BriefSchema, float]:
    """Run generate_brief once; return (brief, elapsed_seconds).

    We use asyncio.run here because pytest-asyncio does not support
    module-scoped async fixtures in all configurations — using asyncio.run
    inside a sync fixture is reliable and avoids scope mismatches.
    """
    import asyncio

    meeting_dt = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    t0 = time.perf_counter()
    brief = asyncio.run(generate_brief("bergstrom", meeting_dt))
    elapsed = time.perf_counter() - t0

    # Persist the cache so the committed file reflects a real run
    save_brief_cache(brief)

    return brief, elapsed


@pytest.fixture(scope="module")
def brief(brief_and_elapsed: tuple[BriefSchema, float]) -> BriefSchema:
    return brief_and_elapsed[0]


@pytest.fixture(scope="module")
def elapsed(brief_and_elapsed: tuple[BriefSchema, float]) -> float:
    return brief_and_elapsed[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_produces_valid_briefschema(brief: BriefSchema) -> None:
    """BriefSchema is returned and round-trips cleanly through model_validate."""
    assert isinstance(brief, BriefSchema), (
        f"generate_brief must return a BriefSchema; got {type(brief)}"
    )
    # Round-trip validation — confirms model_dump JSON is schema-compliant
    roundtripped = BriefSchema.model_validate(brief.model_dump())
    assert roundtripped.client_id == brief.client_id
    assert roundtripped.intel_mode == brief.intel_mode


def test_has_nbas_with_evidence(brief: BriefSchema) -> None:
    """Brief must have 1-3 NBAs, each with ≥1 evidence_ref."""
    assert 1 <= len(brief.three_nbas) <= 3, (
        f"Expected 1-3 NBAs; got {len(brief.three_nbas)}"
    )
    for i, nba in enumerate(brief.three_nbas):
        assert len(nba.evidence_refs) >= 1, (
            f"NBA[{i}] '{nba.title}' has no evidence_refs"
        )


def test_opportunities_present(brief: BriefSchema) -> None:
    """Brief must surface ≥2 opportunity signals (Bergström drift guarantees this)."""
    assert len(brief.opportunities) >= 2, (
        f"Expected ≥2 opportunity signals; got {len(brief.opportunities)}: "
        + str([s.suggested_topic for s in brief.opportunities])
    )


def test_intel_mode_snapshot(brief: BriefSchema) -> None:
    """Intel mode must be 'snapshot' (snapshot-first deployment)."""
    assert brief.intel_mode == "snapshot", (
        f"Expected intel_mode='snapshot'; got '{brief.intel_mode}'"
    )


def test_completes_under_budget(elapsed: float) -> None:
    """Informational timing check — NOT a latency gate.

    The brief is served from a pre-generated cache (cache-first architecture),
    so background generation time is NOT on the user's critical path.  This
    test simply prints the generation cost and asserts a generous sanity ceiling
    (300s) that would only fire on a true hang or catastrophic regression.

    Expected range with restored Pro Planner+Synthesizer + reranker ON:
      ~120–150s on a shared Vertex AI dev project (2–2.5 min).
    """
    print(
        f"\n  brief gen: {elapsed:.1f}s "
        f"(background/pre-generated; cache-first serving — not on request path)"
    )
    assert elapsed < 300.0, (
        f"Brief generation exceeded 300s sanity ceiling (possible hang): {elapsed:.2f}s"
    )
