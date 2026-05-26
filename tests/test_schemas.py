"""Contract tests for the Pydantic schemas."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas import BriefSchema, EvidenceRef, NextBestAction


def _nba(**overrides) -> NextBestAction:
    base = dict(
        title="Trim US tech overweight",
        rationale="US tech runs +5pp over IPS target.",
        projected_impact="Brings allocation back within band.",
        confidence="high",
        evidence_refs=[EvidenceRef(doc_id="bergstrom_ips")],
        suggested_priority="primary",
    )
    base.update(overrides)
    return NextBestAction(**base)


def test_brief_valid_deserializes():
    b = BriefSchema(
        client_id="bergstrom",
        generated_at=datetime(2026, 6, 1, 14, 0),
        intel_mode="snapshot",
        opportunities=[],
        weekend_changes=[],
        three_nbas=[_nba()],
        risk_flags=[],
    )
    assert b.three_nbas[0].suggested_priority == "primary"


def test_brief_missing_required_field_raises():
    with pytest.raises(ValidationError):
        BriefSchema(client_id="bergstrom")  # missing generated_at, intel_mode, ...


def test_nba_without_evidence_refs_raises():
    with pytest.raises(ValidationError):
        NextBestAction(
            title="t",
            rationale="r",
            projected_impact="p",
            confidence="high",
            suggested_priority="primary",
        )  # evidence_refs omitted


def test_brief_rejects_more_than_three_nbas():
    with pytest.raises(ValidationError):
        BriefSchema(
            client_id="bergstrom",
            generated_at=datetime(2026, 6, 1, 14, 0),
            intel_mode="snapshot",
            opportunities=[],
            weekend_changes=[],
            three_nbas=[_nba(), _nba(), _nba(), _nba()],
            risk_flags=[],
        )
