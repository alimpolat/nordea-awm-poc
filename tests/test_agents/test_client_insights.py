"""Live tests for the Stage-2 Client Insights agent.

These tests make a real Vertex AI / Gemini Flash call.
They are marked 'live' and deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (in pyproject.toml) means plain `async def` tests work.
"""
import pytest

from app.agents import client_insights

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
async def snap():
    """Call the agent once; reuse the result across all tests in this module."""
    result = await client_insights.run("bergstrom")
    # Print for CI visibility / human eyeball check
    print("\n--- Client Insights snapshot ---")
    print(f"  client_id={result.client_id!r}  aum_sek={result.aum_sek:,.0f}")
    print(f"  holdings count={len(result.holdings)}")
    print(f"  target_allocation sum={sum(result.target_allocation.values()):.6f}")
    print(f"  last_meeting_date={result.last_meeting_date}")
    print(f"\n  stated_concerns ({len(result.stated_concerns)}):")
    for c in result.stated_concerns:
        print(f"    - {c}")
    print(f"\n  restrictions ({len(result.restrictions)}):")
    for r in result.restrictions:
        safe = r.encode("ascii", errors="replace").decode("ascii")
        print(f"    - {safe}")
    return result


async def test_holdings_count(snap):
    """Fixture has 17 holdings; reconcile step overwrites from JSON so this is deterministic."""
    assert len(snap.holdings) == 17, (
        f"Expected 17 holdings, got {len(snap.holdings)}"
    )


async def test_target_allocation_sums_to_one(snap):
    total = sum(snap.target_allocation.values())
    assert abs(total - 1.0) < 1e-6, (
        f"target_allocation sums to {total:.8f}, expected 1.0"
    )


async def test_stated_concerns_count(snap):
    """The most recent meeting note contains 3 stated concerns."""
    assert len(snap.stated_concerns) >= 2, (
        f"Expected ≥2 stated_concerns, got {len(snap.stated_concerns)}: "
        f"{snap.stated_concerns}"
    )


async def test_restrictions_present(snap):
    assert len(snap.restrictions) >= 1, (
        f"Expected ≥1 restriction, got {len(snap.restrictions)}"
    )


async def test_client_id_and_aum(snap):
    assert snap.client_id == "bergstrom", (
        f"Expected client_id='bergstrom', got {snap.client_id!r}"
    )
    assert snap.aum_sek == 480_000_000, (
        f"Expected aum_sek=480000000, got {snap.aum_sek}"
    )
