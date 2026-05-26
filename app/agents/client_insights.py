"""Stage 2 — Client Insights agent.

Assembles a ClientSnapshot from the Bergström portfolio JSON, IPS, and meeting
notes.  The LLM's genuine value-add is the *interpretive* extraction:
  - ``stated_concerns``  — from the meeting notes
  - ``restrictions``     — from the IPS

All numeric / structured fields (holdings, aum_sek, target_allocation, etc.)
are **overwritten from the authoritative fixture JSON** after the LLM call so
there is no numeric drift from generation.

Design decisions
----------------
* Async interface mirrors opportunity_scout: ``asyncio.to_thread`` wraps the
  blocking Gemini call so the orchestrator can gather Stage-1..4 in parallel.
* ``_build_contents`` gives the model: full IPS text (for restrictions),
  all meeting notes (for stated_concerns), and a lightweight portfolio
  summary for context.  It does NOT ask the model to re-compute numbers.
* The reconcile step at the bottom of ``_run_sync`` is the governance rule
  "numbers computed not generated": it replaces every structured numeric field
  with values read directly from the JSON fixture.
"""
import asyncio
import json
from datetime import date
from typing import Any

from app.agents._base import run_agent_sync
from app.fixtures import load_portfolio, load_ips_text, load_meeting_notes
from app.schemas import ClientSnapshot, Holding


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def run(client_id: str = "bergstrom") -> ClientSnapshot:
    """Async entry point — wraps the blocking LLM call in a thread pool."""
    return await asyncio.to_thread(_run_sync, client_id)


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------

def _run_sync(client_id: str) -> ClientSnapshot:
    pf: dict[str, Any] = load_portfolio(client_id)
    ips: str = load_ips_text(client_id)
    notes: list[str] = load_meeting_notes(client_id)

    contents = _build_contents(pf, ips, notes)
    snap: ClientSnapshot = run_agent_sync("client_insights", contents, ClientSnapshot)

    # Reconcile — overwrite every deterministic field with the authoritative
    # fixture values so the snapshot the brief consumes has exact numbers
    # (no LLM drift on holdings / allocations).  Keep the model's
    # stated_concerns and restrictions.
    snap.holdings = [Holding(**h) for h in pf["holdings"]]
    snap.aum_sek = pf["aum_sek"]
    snap.target_allocation = pf["target_allocation"]
    snap.client_id = pf["client_id"]
    snap.client_name = pf["client_name"]
    snap.last_meeting_date = date.fromisoformat(pf["last_meeting_date"])
    return snap


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _build_contents(pf: dict[str, Any], ips: str, notes: list[str]) -> str:
    """Return the user-turn string sent to the model.

    The model receives:
      1. The full IPS plain text  → to extract restrictions
      2. All meeting notes        → to extract stated_concerns
      3. A brief portfolio summary → for context only (numbers will be overwritten)
    """
    # 1. IPS text
    ips_section = (
        "=== INVESTMENT POLICY STATEMENT (full text) ===\n"
        + ips
    )

    # 2. Meeting notes — label each note with its index so the model can
    #    identify the most recent one (chronologically last in the list).
    note_parts = []
    for i, note in enumerate(notes, start=1):
        label = f"=== MEETING NOTE {i} of {len(notes)}"
        if i == len(notes):
            label += " (MOST RECENT)"
        label += " ==="
        note_parts.append(label + "\n" + note)
    notes_section = "\n\n".join(note_parts)

    # 3. Portfolio summary (lightweight — exact numbers come from the fixture)
    holdings_lines = ["=== PORTFOLIO SUMMARY (for context) ==="]
    holdings_lines.append(
        f"Client: {pf['client_name']} ({pf['client_id']})  |  "
        f"AUM: SEK {pf['aum_sek']:,.0f}  |  "
        f"Last meeting: {pf['last_meeting_date']}"
    )
    holdings_lines.append(f"Holdings ({len(pf['holdings'])} positions):")
    for h in pf["holdings"]:
        holdings_lines.append(
            f"  {h['ticker']:<14}  {h['name']:<40}  "
            f"asset_class={h['asset_class']:<22}  "
            f"MV={h['current_mv']:>12,.0f} SEK  "
            f"fx={h['fx_exposure']}"
        )
    portfolio_section = "\n".join(holdings_lines)

    # 4. Task instruction
    task = (
        "TASK:\n"
        "Using the source documents above, produce a ClientSnapshot.\n\n"
        "For stated_concerns: read ALL meeting notes carefully and list each "
        "distinct concern the client raised (verbatim-in-spirit, using the "
        "client's own framing). The most recent meeting note contains at least "
        "3 concerns — capture all of them.\n\n"
        "For restrictions: read the IPS and list each distinct policy constraint "
        "as a short declarative sentence (e.g. 'Single position ≤ 5% of AUM', "
        "'FX base exposure ≥ 60% SEK', 'No fossil-fuel equities'). Do not merge "
        "separate restrictions into one entry.\n\n"
        "For all other fields (client_id, client_name, aum_sek, holdings, "
        "target_allocation, last_meeting_date): copy the values faithfully from "
        "the portfolio summary — these will be reconciled from the authoritative "
        "source after your call, but the schema requires you to populate them.\n"
        "target_allocation values must be fractional weights (0.30 = 30%), "
        "summing to 1.0."
    )

    return "\n\n".join([ips_section, notes_section, portfolio_section, task])
