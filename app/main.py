"""FastAPI entrypoint. Serves the built React frontend (when present) and the API.

Route ordering: ALL explicit API routes MUST be declared before app.mount("/", ...)
because the StaticFiles catch-all would otherwise shadow them.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import BriefSchema, ChatRequest, ChatResponse, ClientSnapshot, HitlRequest
from app.orchestrator import (
    generate_brief,
    save_brief_cache,
    BRIEF_CACHE_PATH,
    _snapshot_cache_path_for,
)
from app.agents import chat

logger = logging.getLogger(__name__)

app = FastAPI(title="Nordea AWM AI POC")

# ---------------------------------------------------------------------------
# In-memory HITL log (per design §13 "logged in-memory for POC")
# ---------------------------------------------------------------------------
HITL_LOG: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_path_for(client_id: str) -> Path:
    """Return the cache file path for a given client_id.

    For 'bergstrom' this matches BRIEF_CACHE_PATH exactly.
    For other clients it follows the same naming convention.
    """
    if client_id == "bergstrom":
        return BRIEF_CACHE_PATH
    return BRIEF_CACHE_PATH.parent / f"brief_cache_{client_id}.json"


async def _regen_async(client_id: str) -> None:
    """Fire-and-forget coroutine: regenerate the brief and write it to disk.

    Launched via asyncio.create_task so it survives even when the calling
    handler raises an HTTPException (Starlette BackgroundTasks are discarded
    on exception; asyncio tasks are not).  Failures are logged and swallowed
    so they never crash the server.
    """
    try:
        brief = await generate_brief(client_id, datetime.now(timezone.utc))
        save_brief_cache(brief, _cache_path_for(client_id))
        logger.info("Background regen complete for %s", client_id)
    except Exception:
        logger.exception("background brief regeneration failed for %s", client_id)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Brief (cache-first)
# ---------------------------------------------------------------------------


@app.get("/api/brief/{client_id}", response_model=BriefSchema)
async def get_brief(client_id: str, refresh: bool = False):
    """Return the pre-generated brief from cache (instant, zero network).

    Plain GET: serves the committed cache immediately — no LLM call, no
    network activity.  If the cache is missing, returns 503.

    ?refresh=true: fires a best-effort background regeneration on the event
    loop via asyncio.create_task (survives even when this handler later
    raises).  The task runs the full ~2-minute pipeline off the request path.
    - If a cache already exists, it is served immediately (200) while regen
      runs in the background.
    - If no cache exists yet, returns 202 telling the caller to retry shortly
      once the background task has completed.
    """
    cache_path = _cache_path_for(client_id)

    # Fire background regen on the event loop BEFORE any potential raise.
    # asyncio.create_task survives a subsequent HTTPException; Starlette
    # BackgroundTasks (the previous implementation) do not.
    if refresh:
        asyncio.create_task(_regen_async(client_id))

    if cache_path.exists():
        return BriefSchema.model_validate_json(
            cache_path.read_text(encoding="utf-8")
        )

    if refresh:
        # Regen was just kicked off; tell the caller to retry shortly.
        raise HTTPException(
            status_code=202,
            detail=(
                f"Brief for '{client_id}' is being generated; retry shortly."
            ),
        )

    # No cache and no refresh requested
    raise HTTPException(
        status_code=503,
        detail=(
            f"Brief for '{client_id}' not generated yet. "
            "Run `python -m app.orchestrator` to build the cache "
            "or call with ?refresh=true and retry shortly."
        ),
    )


# ---------------------------------------------------------------------------
# Client snapshot (cache-first)
# ---------------------------------------------------------------------------


@app.get("/api/client/{client_id}", response_model=ClientSnapshot)
async def get_client(client_id: str):
    """Return the pre-generated ClientSnapshot from cache (instant, zero network).

    Serves client name, AUM, holdings, and target allocations for the
    frontend cockpit (ClientHeader, AllocationDonut, holdings table).
    Cache is written by the orchestrator as a side-effect of Stage 2.
    """
    path = _snapshot_cache_path_for(client_id)
    if path.exists():
        return ClientSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    raise HTTPException(
        status_code=503,
        detail=(
            f"Client snapshot for '{client_id}' not generated yet. "
            "Run `python -m app.orchestrator` to build the cache."
        ),
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest):
    """Answer a question about the client's brief / portfolio using hybrid retrieval."""
    return await chat.run(req)


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------


@app.post("/api/hitl/{action}")
async def post_hitl(
    action: Literal["approve", "edit", "regenerate", "reject"],
    req: HitlRequest,
):
    """Log a human-in-the-loop action against an NBA recommendation."""
    entry = {
        "action": action,
        "client_id": req.client_id,
        "nba_title": req.nba_title,
        "nba_index": req.nba_index,
        "reason": req.reason,
        "edited_text": req.edited_text,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    HITL_LOG.append(entry)
    return {"ok": True, "action": action, "logged": entry, "log_size": len(HITL_LOG)}


# ---------------------------------------------------------------------------
# Static frontend — mounted LAST so it never shadows the API routes above
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:

    @app.get("/")
    def root():
        return HTMLResponse("<h1>Nordea AWM AI POC — frontend not built yet</h1>")
