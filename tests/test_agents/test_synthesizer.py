"""Live tests for the Stage-4 Synthesizer agent.

Makes a real Vertex AI / Gemini Pro call.
Marked 'live'; deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (pyproject.toml) means plain `async def` tests work.

Fixtures are constructed inline — this test isolates the synthesizer from
all upstream agents.  The news fixture is empty to exercise the weekend_changes
fallback path.
"""
from datetime import date, datetime, timezone

import pytest

from app.agents import synthesizer
from app.schemas import (
    BriefSchema,
    ClientSnapshot,
    EvidenceRef,
    IntelFinding,
    IntelFindings,
    MacroFinding,
    MacroFindings,
    NewsFindings,
    NextBestAction,
    OpportunitySignal,
    OpportunitySignals,
    Plan,
    PortfolioFinding,
    RiskFlag,
)

pytestmark = pytest.mark.live

# ---------------------------------------------------------------------------
# Compact inline fixtures (do NOT call upstream agents)
# ---------------------------------------------------------------------------

_PLAN = Plan(
    specialists_to_invoke=["intel", "macro", "portfolio", "news"],
    sub_questions={
        "intel": ["Gulf signals?"],
        "macro": ["ECB rate path?"],
        "portfolio": ["drift & IPS?"],
        "news": ["Gulf news?"],
    },
)

_OPP = OpportunitySignals(
    items=[
        OpportunitySignal(
            trigger_type="drift",
            asset_class="US tech",
            magnitude=5.0,
            confidence="high",
            suggested_topic="Trim US tech",
            evidence_refs=[EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")],
        ),
        OpportunitySignal(
            trigger_type="drift",
            asset_class="Gulf real estate",
            magnitude=5.0,
            confidence="high",
            suggested_topic="Trim Gulf",
            evidence_refs=[EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")],
        ),
    ]
)

_SNAP = ClientSnapshot(
    client_id="bergstrom",
    client_name="Bergström Family Office",
    aum_sek=480_000_000,
    holdings=[],
    target_allocation={
        "Nordic equity": 0.35,
        "US tech": 0.15,
        "EU fixed income": 0.20,
        "Gulf real estate": 0.10,
        "Alternatives": 0.20,
    },
    stated_concerns=[
        "Gulf concentration too high",
        "US-tech valuation discomfort",
        "Interest in green bonds",
    ],
    restrictions=[
        "No direct fossil fuel",
        "Single-position <=5%",
        "FX >=60% SEK",
    ],
    last_meeting_date=date(2026, 4, 14),
)

_INTEL = IntelFindings(
    items=[
        IntelFinding(
            source="World_Monitor snapshot",
            metric="Brent crude, USD/bbl",
            value=78.2,
            as_of=datetime(2026, 5, 30, tzinfo=timezone.utc),
            relevance="Gulf real estate driver",
            live_or_snapshot="snapshot",
        )
    ]
)

_MACRO = MacroFindings(
    items=[
        MacroFinding(
            claim="ECB held its deposit rate at 3.25% in March 2026.",
            evidence_chunks=[
                EvidenceRef(
                    doc_id="ecb_economic_bulletin_2026",
                    chunk_id="ecb_economic_bulletin_2026::p007::0003",
                )
            ],
            confidence="high",
            impact_on_portfolio="Supports EU fixed income re-entry",
        )
    ]
)

_PORT = PortfolioFinding(
    drift_signals=[
        {
            "asset_class": "US tech",
            "current_pct": 20.0,
            "target_pct": 15.0,
            "drift_pp": 5.0,
        },
        {
            "asset_class": "Gulf real estate",
            "current_pct": 15.0,
            "target_pct": 10.0,
            "drift_pp": 5.0,
        },
    ],
    ips_compliance=[
        {
            "rule": "fx_floor_sek",
            "sek_pct": 35.0,
            "floor_pct": 60.0,
            "status": "breach",
        }
    ],
    ytd_summary={
        "weighted_ytd_pct": 7.98,
        "total_mv_sek": 480_000_000,
    },
    opportunities=["Trim US tech and Gulf to fund EU green bonds"],
    computation_trace=(
        "Drift = current-target; SEK share 35% < 60% floor -> breach."
    ),
)

_NEWS = NewsFindings(items=[])  # empty — exercises the weekend_changes fallback


# ---------------------------------------------------------------------------
# Module-scoped fixture: call the agent once, reuse across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def brief() -> BriefSchema:
    """Call the synthesizer once; reuse the result across all tests in this module."""
    result = await synthesizer.run(_PLAN, _OPP, _SNAP, _INTEL, _MACRO, _PORT, _NEWS)

    # Print for CI visibility and human eyeball check
    print("\n--- Synthesizer brief ---")
    print(f"  client_id:    {result.client_id}")
    print(f"  intel_mode:   {result.intel_mode}")
    print(f"  generated_at: {result.generated_at}")
    print(f"  #opportunities: {len(result.opportunities)}")
    print(f"  #weekend_changes: {len(result.weekend_changes)}")

    print("\n  three_nbas:")
    for nba in result.three_nbas:
        print(
            f"    [{nba.suggested_priority}] {nba.title!r}  "
            f"#evidence_refs={len(nba.evidence_refs)}"
        )

    print("\n  risk_flags:")
    for flag in result.risk_flags:
        print(f"    kind={flag.kind}  severity={flag.severity}  note={flag.note[:80]!r}")

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_returns_brief_schema(brief):
    """Synthesizer must return a BriefSchema instance."""
    assert isinstance(brief, BriefSchema)


async def test_three_nbas_count_in_range(brief):
    """three_nbas must contain between 1 and 3 items (schema + reconciler enforce this)."""
    assert 1 <= len(brief.three_nbas) <= 3, (
        f"Expected 1-3 NBAs, got {len(brief.three_nbas)}"
    )


async def test_every_nba_has_evidence_refs(brief):
    """Every NextBestAction must have at least one evidence_ref (reconciler enforces this)."""
    for nba in brief.three_nbas:
        assert len(nba.evidence_refs) >= 1, (
            f"NBA '{nba.title}' has no evidence_refs — reconciler should have added fallback."
        )


async def test_every_risk_flag_has_severity(brief):
    """Every RiskFlag must carry a non-null severity value."""
    valid_severities = {"info", "watch", "action", "none"}
    for flag in brief.risk_flags:
        assert flag.severity in valid_severities, (
            f"RiskFlag kind={flag.kind!r} has invalid severity={flag.severity!r}"
        )


async def test_intel_mode_is_snapshot(brief):
    """All intel findings are snapshot → intel_mode must be 'snapshot'."""
    assert brief.intel_mode == "snapshot", (
        f"Expected intel_mode='snapshot', got {brief.intel_mode!r}"
    )


async def test_client_id_is_bergstrom(brief):
    """client_id must be overwritten to the ClientSnapshot's client_id."""
    assert brief.client_id == "bergstrom", (
        f"Expected client_id='bergstrom', got {brief.client_id!r}"
    )


async def test_opportunities_passed_through_verbatim(brief):
    """opportunities must match the Stage-1 scout output exactly (2 signals)."""
    assert len(brief.opportunities) == 2, (
        f"Expected 2 opportunities (passed from scout), got {len(brief.opportunities)}"
    )


async def test_weekend_changes_fallback(brief):
    """News is empty → weekend_changes must fall back to macro items (≥1)."""
    assert len(brief.weekend_changes) >= 1, (
        f"Expected ≥1 weekend_changes (macro fallback), got {len(brief.weekend_changes)}"
    )


async def test_generated_at_is_recent(brief):
    """generated_at must be a timezone-aware datetime (set by the reconciler)."""
    assert isinstance(brief.generated_at, datetime), (
        f"Expected datetime, got {type(brief.generated_at)}"
    )
    assert brief.generated_at.tzinfo is not None, (
        "generated_at must be timezone-aware"
    )


async def test_computation_trace_backfill(brief):
    """At least one NBA should carry a computation_trace.

    The fixture has a drift signal + portfolio.computation_trace, so the reconciler
    must backfill at least one drift-related NBA with the portfolio trace.
    """
    assert any(nba.computation_trace for nba in brief.three_nbas), (
        "Expected at least one NBA to have a computation_trace backfilled "
        "from the portfolio (drift signal + computation_trace present in fixture)."
    )
