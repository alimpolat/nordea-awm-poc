"""API route tests for app/main.py.

Offline tests (no marker): healthz, brief cache-first, 503 for unknown client,
HITL log, 422 for invalid HITL action.

Live tests (@pytest.mark.live): chat route with a real Gemini call.

Run offline only:
    uv run --no-sync pytest tests/test_api.py -m "not live" -q

Run live only:
    uv run --no-sync pytest tests/test_api.py -m "live" -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, HITL_LOG
from app.schemas import BriefSchema, ChatResponse, ClientSnapshot

client = TestClient(app)


# ---------------------------------------------------------------------------
# OFFLINE tests — zero network calls
# ---------------------------------------------------------------------------


def test_healthz():
    """GET /healthz should return 200 with {ok: true}."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_get_brief_serves_cache():
    """GET /api/brief/bergstrom (no refresh) serves the committed cache instantly.

    This test makes zero network calls:
    - refresh is NOT passed (no background regen scheduled)
    - the cache file is pre-committed and read directly from disk
    """
    resp = client.get("/api/brief/bergstrom")
    assert resp.status_code == 200, f"Expected 200; got {resp.status_code}: {resp.text}"

    body = resp.json()
    brief = BriefSchema.model_validate(body)

    assert brief.intel_mode == "snapshot", (
        f"Expected intel_mode='snapshot'; got {brief.intel_mode!r}"
    )
    assert 1 <= len(brief.three_nbas) <= 3, (
        f"Expected 1-3 NBAs; got {len(brief.three_nbas)}"
    )


def test_get_brief_unknown_client_503():
    """GET /api/brief/nonexistent should return 503 (no cache file exists)."""
    resp = client.get("/api/brief/nonexistent_client_xyz")
    assert resp.status_code == 503, (
        f"Expected 503 for unknown client; got {resp.status_code}"
    )
    assert "not generated yet" in resp.json()["detail"].lower()


def test_hitl_logs():
    """POST /api/hitl/approve and /api/hitl/reject should log entries and return ok."""
    # Clear any state from previous test runs in this session
    initial_size = len(HITL_LOG)

    # First action: approve
    resp1 = client.post(
        "/api/hitl/approve",
        json={
            "client_id": "bergstrom",
            "nba_title": "Reduce US Tech overweight",
            "nba_index": 0,
        },
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["ok"] is True
    assert data1["action"] == "approve"
    assert data1["log_size"] == initial_size + 1

    # Second action: reject with reason
    resp2 = client.post(
        "/api/hitl/reject",
        json={
            "client_id": "bergstrom",
            "nba_title": "Reduce US Tech overweight",
            "nba_index": 0,
            "reason": "Client disagrees with timing",
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["ok"] is True
    assert data2["action"] == "reject"
    assert data2["log_size"] == initial_size + 2, (
        f"Expected log_size to increment to {initial_size + 2}; got {data2['log_size']}"
    )
    assert data2["logged"]["reason"] == "Client disagrees with timing"


def test_get_client_serves_cache():
    """GET /api/client/bergstrom serves the committed snapshot cache (zero network).

    Validates ClientSnapshot shape, checks 17 holdings, aum 480000000, and
    client_id == 'bergstrom'.
    """
    resp = client.get("/api/client/bergstrom")
    assert resp.status_code == 200, f"Expected 200; got {resp.status_code}: {resp.text}"

    body = resp.json()
    snap = ClientSnapshot.model_validate(body)

    assert snap.client_id == "bergstrom", f"Expected client_id='bergstrom'; got {snap.client_id!r}"
    assert len(snap.holdings) == 17, f"Expected 17 holdings; got {len(snap.holdings)}"
    assert snap.aum_sek == 480_000_000, f"Expected aum_sek=480000000; got {snap.aum_sek}"


def test_get_client_unknown_503():
    """GET /api/client/nonexistent should return 503 (no snapshot cache exists)."""
    resp = client.get("/api/client/nonexistent_client_xyz")
    assert resp.status_code == 503, (
        f"Expected 503 for unknown client; got {resp.status_code}"
    )
    assert "not generated yet" in resp.json()["detail"].lower()


def test_hitl_invalid_action_422():
    """POST /api/hitl/frobnicate should return 422 (not in the Literal enum)."""
    resp = client.post(
        "/api/hitl/frobnicate",
        json={"client_id": "bergstrom"},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for invalid HITL action; got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# LIVE tests — require Vertex AI credentials
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_chat_route_live():
    """POST /api/chat with a real Bergström question — validates ChatResponse shape.

    Asserts:
    - 200 response
    - answer is non-empty
    - cited_refs is non-empty (minimal chat populates from retrieval fallback)
    """
    resp = client.post(
        "/api/chat",
        json={
            "client_id": "bergstrom",
            "question": "What is Bergström's Gulf real estate exposure?",
        },
    )
    assert resp.status_code == 200, (
        f"Expected 200 from chat route; got {resp.status_code}: {resp.text}"
    )

    body = resp.json()
    chat_resp = ChatResponse.model_validate(body)

    assert chat_resp.answer, "ChatResponse.answer must be non-empty"
    assert chat_resp.cited_refs, (
        "ChatResponse.cited_refs must be non-empty "
        "(minimal chat populates from top retrieval chunks)"
    )

    # Print the response for the build report
    print(f"\n  answer: {chat_resp.answer[:300]}{'...' if len(chat_resp.answer) > 300 else ''}")
    print(f"  cited_refs ({len(chat_resp.cited_refs)}):")
    for ref in chat_resp.cited_refs:
        print(f"    doc_id={ref.doc_id!r}  chunk_id={ref.chunk_id!r}")
    print(f"  confidence: {chat_resp.confidence}")
