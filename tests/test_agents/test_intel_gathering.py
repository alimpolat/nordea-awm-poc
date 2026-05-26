"""Live tests for the Stage-3b Intel Gathering specialist.

Makes a real Vertex AI / Gemini Flash call.
Marked 'live'; deselected from the offline suite with -m 'not live'.
pytest asyncio_mode = "auto" (pyproject.toml) means plain `async def` tests work.

Shape-based asserts only — robust to which signals the model selects.
"""
import pytest

from app.agents import intel_gathering

pytestmark = pytest.mark.live

_SUB_QUESTIONS = [
    "What Gulf real-estate and oil signals are most relevant to the Bergström portfolio this week?",
    "Are there any Nordic equity or FX signals that could affect the client's SEK base exposure?",
    "What ECB or central bank signals should the advisor discuss at the Monday meeting?",
]


@pytest.fixture(scope="module")
async def intel_findings():
    """Call the agent once; reuse across all tests in this module."""
    result = await intel_gathering.run(_SUB_QUESTIONS)

    # Print for CI visibility and human eyeball check
    print("\n--- Intel Gathering findings ---")
    for item in result.items:
        print(
            f"  metric={item.metric!r}  "
            f"value={item.value!r}  "
            f"live_or_snapshot={item.live_or_snapshot!r}"
        )
    return result


async def test_returns_at_least_one_item(intel_findings):
    """Reconciliation must leave at least one item (or fall back to full ground-truth)."""
    assert len(intel_findings.items) >= 1, (
        f"Expected ≥1 intel items, got {len(intel_findings.items)}"
    )


async def test_all_items_are_snapshot(intel_findings):
    """LIVE_ROUTES is empty in the POC — every item must be tagged 'snapshot'."""
    for item in intel_findings.items:
        assert item.live_or_snapshot == "snapshot", (
            f"Expected live_or_snapshot='snapshot' (LIVE_ROUTES is empty), "
            f"got {item.live_or_snapshot!r} for metric={item.metric!r}"
        )


async def test_live_or_snapshot_field_valid(intel_findings):
    """Every item must carry a valid live_or_snapshot tag."""
    valid_tags = {"live", "snapshot"}
    for item in intel_findings.items:
        assert item.live_or_snapshot in valid_tags, (
            f"Invalid live_or_snapshot={item.live_or_snapshot!r} for metric={item.metric!r}"
        )


async def test_items_have_non_empty_metric_and_value(intel_findings):
    """Every item must have a non-empty metric string and a non-empty value."""
    for item in intel_findings.items:
        assert isinstance(item.metric, str) and item.metric.strip(), (
            f"Item has empty metric: {item!r}"
        )
        # value is str | float — check it's not None and, if str, not blank
        assert item.value is not None, f"Item has None value: {item!r}"
        if isinstance(item.value, str):
            assert item.value.strip(), f"Item has blank string value: {item!r}"


async def test_items_are_ground_truth_metrics(intel_findings):
    """All returned metrics must be drawn from the World Monitor snapshot
    (reconciliation should have stripped any fabricated metric names)."""
    from app.intel.world_monitor_client import fetch_intel
    ground_truth = fetch_intel()
    gt_metrics_lower = {item.metric.lower() for item in ground_truth.items}

    for item in intel_findings.items:
        item_metric_lower = item.metric.lower()
        # Accept if the returned metric is a substring of any gt metric or vice-versa
        matched = any(
            item_metric_lower in gt_lower or gt_lower in item_metric_lower
            for gt_lower in gt_metrics_lower
        )
        assert matched, (
            f"Returned metric {item.metric!r} does not match any ground-truth metric. "
            f"Ground-truth metrics: {sorted(gt_metrics_lower)}"
        )
