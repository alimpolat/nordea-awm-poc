"""Validate the Bergström portfolio fixture: schema, totals, allocation arithmetic."""
import json
from pathlib import Path

from app.schemas import Holding

DATA = Path(__file__).resolve().parents[1] / "data" / "bergstrom_portfolio.json"


def _load():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_holdings_parse_as_schema():
    holdings = [Holding(**h) for h in _load()["holdings"]]
    assert len(holdings) >= 15


def test_total_mv_equals_aum_480m():
    d = _load()
    total = sum(h["current_mv"] for h in d["holdings"])
    assert total == d["aum_sek"] == 480_000_000


def test_category_sums_match_current_allocation():
    d = _load()
    total = d["aum_sek"]
    by_cat: dict[str, int] = {}
    for h in d["holdings"]:
        by_cat[h["asset_class"]] = by_cat.get(h["asset_class"], 0) + h["current_mv"]
    for cat, weight in d["current_allocation"].items():
        assert abs(by_cat[cat] / total - weight) < 1e-3, (cat, by_cat[cat] / total, weight)


def test_allocations_sum_to_one():
    d = _load()
    assert abs(sum(d["target_allocation"].values()) - 1.0) < 1e-9
    assert abs(sum(d["current_allocation"].values()) - 1.0) < 1e-9


def test_gulf_and_ustech_are_overweight_vs_target():
    """The drift the Opportunity-Scout must detect."""
    d = _load()
    cur, tgt = d["current_allocation"], d["target_allocation"]
    assert abs((cur["US tech"] - tgt["US tech"]) - 0.05) < 1e-9
    assert abs((cur["Gulf real estate"] - tgt["Gulf real estate"]) - 0.05) < 1e-9
