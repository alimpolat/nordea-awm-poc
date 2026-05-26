"""Offline unit tests for the Stage 3a Planner invariant.

No live marker — runs in the default offline suite (-m 'not live').
Monkeypatches run_agent_sync in the planner module so NO network call is made.

Invariant under test
--------------------
After _run_sync() returns, regardless of what the model emitted:
  1. specialists_to_invoke == all four (in any order).
  2. Every specialist key in sub_questions has at least one non-blank question.
  3. output_schema_name == "BriefSchema".
  4. A "real" question already present for a specialist is preserved (not clobbered).
"""
import asyncio
from datetime import date, datetime

import pytest

from app.agents import planner as planner_module
from app.agents.planner import _run_sync, _ALL_SPECIALISTS, _FALLBACK_QUESTIONS
from app.schemas import (
    ClientSnapshot,
    EvidenceRef,
    OpportunitySignal,
    OpportunitySignals,
    Plan,
)


# ---------------------------------------------------------------------------
# Minimal input fixtures (same shape as the live test; no network needed)
# ---------------------------------------------------------------------------

def _make_sigs() -> OpportunitySignals:
    return OpportunitySignals(items=[
        OpportunitySignal(
            trigger_type="drift",
            asset_class="US tech",
            magnitude=5.0,
            confidence="high",
            suggested_topic="Trim US tech overweight",
            evidence_refs=[EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")],
        ),
    ])


def _make_snap() -> ClientSnapshot:
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
        stated_concerns=["Gulf concentration too high"],
        restrictions=["No direct fossil fuel"],
        last_meeting_date=date(2026, 4, 14),
    )


_MEETING_DT = datetime(2026, 6, 1, 14, 0)


# ---------------------------------------------------------------------------
# Case 1 — partial / blank model output
#   "macro" has a real question; "intel" has only blanks; "news" is absent;
#   "portfolio" key missing entirely.  output_schema_name is empty.
# ---------------------------------------------------------------------------

def test_invariant_repairs_partial_blank_output(monkeypatch):
    """Blank / missing specialists get fallback; the real macro question survives."""
    degenerate_plan = Plan(
        specialists_to_invoke=["macro"],
        sub_questions={
            "macro": ["What is the macro outlook?"],
            "intel": [""],           # blank-only list → should be replaced
            "news": [],              # empty list      → should be replaced
            # "portfolio" is absent  → should be filled with fallback
        },
        output_schema_name="",
    )

    monkeypatch.setattr(
        "app.agents.planner.run_agent_sync",
        lambda *a, **k: degenerate_plan,
    )

    result = _run_sync(_make_sigs(), _make_snap(), _MEETING_DT)

    # 1. All four specialists must be in specialists_to_invoke
    assert set(result.specialists_to_invoke) == set(_ALL_SPECIALISTS), (
        f"specialists_to_invoke mismatch: {result.specialists_to_invoke}"
    )

    # 2. Every specialist has at least one non-blank question
    for specialist in _ALL_SPECIALISTS:
        assert specialist in result.sub_questions, (
            f"Missing sub_questions key: {specialist!r}"
        )
        questions = result.sub_questions[specialist]
        assert any(q.strip() for q in questions), (
            f"sub_questions[{specialist!r}] has no non-blank question: {questions!r}"
        )

    # 3. output_schema_name must be BriefSchema
    assert result.output_schema_name == "BriefSchema"

    # 4. The original real macro question is preserved (not replaced by fallback)
    assert "What is the macro outlook?" in result.sub_questions["macro"], (
        f"Real macro question was clobbered: {result.sub_questions['macro']!r}"
    )

    # 5. Blank "intel" entry must have been replaced with fallback (no blanks survive)
    for q in result.sub_questions["intel"]:
        assert q.strip(), f"Blank question survived in intel: {q!r}"

    # 6. "intel" fallback is the canonical fallback text
    assert result.sub_questions["intel"] == _FALLBACK_QUESTIONS["intel"]

    # 7. "news" (empty list) gets fallback
    assert result.sub_questions["news"] == _FALLBACK_QUESTIONS["news"]

    # 8. "portfolio" (absent) gets fallback
    assert result.sub_questions["portfolio"] == _FALLBACK_QUESTIONS["portfolio"]


# ---------------------------------------------------------------------------
# Case 2 — all sub_questions completely absent
# ---------------------------------------------------------------------------

def test_invariant_repairs_all_missing(monkeypatch):
    """When sub_questions is entirely empty, every specialist gets fallback questions."""
    empty_plan = Plan(
        specialists_to_invoke=[],
        sub_questions={},
        output_schema_name="",
    )

    monkeypatch.setattr(
        "app.agents.planner.run_agent_sync",
        lambda *a, **k: empty_plan,
    )

    result = _run_sync(_make_sigs(), _make_snap(), _MEETING_DT)

    assert set(result.specialists_to_invoke) == set(_ALL_SPECIALISTS)
    assert result.output_schema_name == "BriefSchema"

    for specialist in _ALL_SPECIALISTS:
        assert specialist in result.sub_questions
        assert result.sub_questions[specialist] == _FALLBACK_QUESTIONS[specialist], (
            f"Expected fallback for {specialist!r}, got: {result.sub_questions[specialist]!r}"
        )
