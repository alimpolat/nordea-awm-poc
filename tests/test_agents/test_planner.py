"""Live test for the Stage 3a Planner agent.

Makes a real Vertex AI / Gemini Pro call.
Marked 'live'; deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (pyproject.toml) means plain `async def` tests work.

Inputs are constructed inline (minimal fixtures) — we deliberately do NOT call
the scout or insights agents to keep this test cheap and focused.
"""
import pytest
from datetime import date, datetime

from app.schemas import (
    ClientSnapshot,
    EvidenceRef,
    OpportunitySignal,
    OpportunitySignals,
)
from app.agents import planner

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sigs() -> OpportunitySignals:
    return OpportunitySignals(items=[
        OpportunitySignal(
            trigger_type="drift",
            asset_class="US tech",
            magnitude=5.0,
            confidence="high",
            suggested_topic="Trim US tech overweight",
            evidence_refs=[EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")],
        ),
        OpportunitySignal(
            trigger_type="drift",
            asset_class="Gulf real estate",
            magnitude=5.0,
            confidence="high",
            suggested_topic="Discuss Gulf concentration",
            evidence_refs=[EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")],
        ),
    ])


@pytest.fixture(scope="module")
def snap() -> ClientSnapshot:
    return ClientSnapshot(
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


@pytest.fixture(scope="module")
async def plan(sigs, snap):
    """Run the Planner once and reuse across all tests in this module."""
    result = await planner.run(sigs, snap, datetime(2026, 6, 1, 14, 0))

    # Print sub_questions for human eyeball check
    print("\n--- Planner sub_questions ---")
    for specialist, questions in result.sub_questions.items():
        print(f"\n  [{specialist}]")
        for q in questions:
            # Encode to ASCII for safe terminal output on Windows
            safe_q = q.encode("ascii", errors="replace").decode("ascii")
            print(f"    - {safe_q}")
    print(f"\n  specialists_to_invoke: {result.specialists_to_invoke}")
    print(f"  output_schema_name: {result.output_schema_name}")

    return result


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

async def test_all_four_specialists_invoked(plan):
    """The POC invariant: all four specialists must always be invoked."""
    assert set(plan.specialists_to_invoke) == {"intel", "macro", "portfolio", "news"}, (
        f"Expected all 4 specialists, got: {plan.specialists_to_invoke}"
    )


async def test_sub_questions_cover_all_specialists(plan):
    """Every specialist must have a non-empty sub_questions list."""
    for specialist in ("intel", "macro", "portfolio", "news"):
        assert specialist in plan.sub_questions, (
            f"Missing sub_questions key for specialist: {specialist!r}"
        )
        questions = plan.sub_questions[specialist]
        assert len(questions) >= 1, (
            f"sub_questions[{specialist!r}] is empty"
        )


async def test_output_schema_name(plan):
    """output_schema_name must be 'BriefSchema'."""
    assert plan.output_schema_name == "BriefSchema", (
        f"Expected output_schema_name='BriefSchema', got {plan.output_schema_name!r}"
    )
