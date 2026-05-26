"""Live tests for the Stage-1 Opportunity Scout agent.

These tests make a real Vertex AI / Gemini Flash call.
They are marked 'live' and deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (in pyproject.toml) means plain `async def` tests work.
"""
import pytest

from app.agents import opportunity_scout

pytestmark = pytest.mark.live

VALID_TRIGGER_TYPES = {"drift", "macro", "event", "ips_violation"}
KNOWN_DOC_IDS = {"bergstrom_portfolio_q1_2026", "bergstrom_ips", "world_monitor_snapshot"}


@pytest.fixture(scope="module")
async def signals():
    """Call the agent once; reuse the result across all tests in this module."""
    result = await opportunity_scout.run("bergstrom")
    # Print for CI visibility / human eyeball check
    print("\n--- Opportunity Scout signals ---")
    for s in result.items:
        print(
            f"  trigger={s.trigger_type}  asset_class={s.asset_class!r}  "
            f"magnitude={s.magnitude}  topic={s.suggested_topic!r}  "
            f"evidence_refs={[e.doc_id for e in s.evidence_refs]}"
        )
    return result


async def test_returns_at_least_two_signals(signals):
    assert len(signals.items) >= 2, (
        f"Expected ≥2 signals, got {len(signals.items)}: {signals.items}"
    )


async def test_signals_have_valid_trigger_and_evidence(signals):
    for signal in signals.items:
        assert signal.trigger_type in VALID_TRIGGER_TYPES, (
            f"Invalid trigger_type {signal.trigger_type!r} for signal {signal}"
        )
        assert len(signal.evidence_refs) > 0, (
            f"Signal has no evidence_refs: {signal}"
        )
        # Each evidence_ref must carry a non-empty doc_id string
        for ref in signal.evidence_refs:
            assert isinstance(ref.doc_id, str) and ref.doc_id.strip(), (
                f"evidence_ref has empty doc_id in signal {signal}"
            )

    # At least one signal must cite a doc_id from the known set
    all_doc_ids = {ref.doc_id for s in signals.items for ref in s.evidence_refs}
    assert all_doc_ids & KNOWN_DOC_IDS, (
        f"No signal cites a known doc_id. Got: {all_doc_ids}. Expected one of: {KNOWN_DOC_IDS}"
    )


async def test_surfaces_overweight_drift(signals):
    """Both mandated overweights (Gulf real estate AND US tech) must appear as signals."""

    def _mentions_gulf(s) -> bool:
        combined = (s.asset_class + " " + s.suggested_topic).lower()
        return "gulf" in combined or "real estate" in combined

    def _mentions_us_tech(s) -> bool:
        combined = (s.asset_class + " " + s.suggested_topic).lower()
        # Match "us tech", "us technology", "u.s. tech", or asset_class "US tech" (case-insensitive).
        # Avoid matching "technical" on its own — require a "us" / "u.s" qualifier OR
        # the asset_class field to be exactly "US tech".
        return (
            "us tech" in combined
            or "u.s. tech" in combined
            or "us technology" in combined
            or s.asset_class.lower() == "us tech"
        )

    gulf_hits = [s for s in signals.items if _mentions_gulf(s)]
    us_tech_hits = [s for s in signals.items if _mentions_us_tech(s)]

    assert len(gulf_hits) >= 1, (
        "No signal references Gulf real estate overweight. "
        f"Signals: {[(s.asset_class, s.suggested_topic) for s in signals.items]}"
    )
    assert len(us_tech_hits) >= 1, (
        "No signal references US tech overweight. "
        f"Signals: {[(s.asset_class, s.suggested_topic) for s in signals.items]}"
    )
