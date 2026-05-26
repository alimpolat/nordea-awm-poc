"""Tests for the World_Monitor client + snapshot fallback (offline)."""
import app.intel.world_monitor_client as wm
from app.schemas import IntelFindings


def test_snapshot_mode_returns_all_snapshot_tagged(monkeypatch):
    monkeypatch.setattr(wm.settings, "intel_mode", "snapshot")
    fi = wm.fetch_intel()
    assert isinstance(fi, IntelFindings)
    assert len(fi.items) >= 6
    assert all(f.live_or_snapshot == "snapshot" for f in fi.items)
    # snapshot carries real numbers, not placeholders
    omx = next(f for f in fi.items if "OMXS30" in f.metric)
    assert isinstance(omx.value, (int, float))


def test_live_mode_falls_back_to_snapshot_on_failure(monkeypatch):
    # Inject a live route, then force the live fetch to fail → must fall back, not crash.
    monkeypatch.setattr(wm.settings, "intel_mode", "live")
    monkeypatch.setattr(wm, "LIVE_ROUTES", {"omxs30": "/api/stock-index?symbols=^OMX"})

    def _boom(*a, **k):
        raise RuntimeError("simulated 500")

    monkeypatch.setattr(wm.httpx, "get", _boom)
    fi = wm.fetch_intel()
    assert all(f.live_or_snapshot == "snapshot" for f in fi.items)
