"""Stage 1 — Opportunity Scout agent.

Given the Bergström portfolio, IPS limits, and live/snapshot market intel,
returns ≥2 OpportunitySignals. The two guaranteed signals are the US-tech and
Gulf real-estate +5 pp overweights; additional IPS-violation or macro/event
signals are also expected based on the data.

Design decisions
----------------
* Drift is computed **deterministically in Python** (not by the model) to avoid
  the conflict between code_execution and response_schema structured output.
* The async interface uses asyncio.to_thread so the orchestrator can later
  gather all Stage-1..4 specialists in parallel without blocking the event loop.
* FX floor and single-position limit are read from the portfolio JSON so the
  agent adapts automatically if client mandates change.
"""
import asyncio
from typing import Any

from app.agents._base import run_agent_sync
from app.fixtures import load_portfolio
from app.intel.world_monitor_client import fetch_intel
from app.schemas import IntelFindings, OpportunitySignals


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def run(client_id: str = "bergstrom") -> OpportunitySignals:
    """Async entry point — wraps the blocking LLM call in a thread pool."""
    return await asyncio.to_thread(_run_sync, client_id)


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------

def _run_sync(client_id: str) -> OpportunitySignals:
    pf: dict[str, Any] = load_portfolio(client_id)
    contents = _build_contents(pf)
    return run_agent_sync("opportunity_scout", contents, OpportunitySignals)


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _build_contents(pf: dict[str, Any]) -> str:
    """Return the user-turn string: drift table + IPS limits + holdings + intel."""
    aum = pf["aum_sek"]
    target = pf["target_allocation"]
    current = pf["current_allocation"]

    # 1. Deterministic drift table — iterate over the UNION of target + current asset classes
    #    so classes present in current but missing from target are not silently dropped.
    drift_lines = ["ALLOCATION DRIFT TABLE (asset class → current%, target%, drift pp):"]
    for ac in sorted(set(target) | set(current)):
        cur_pct = current.get(ac, 0.0) * 100
        tgt_pct = target.get(ac, 0.0) * 100
        drift_pp = cur_pct - tgt_pct
        flag = " ← OVERWEIGHT" if drift_pp > 0 else (" ← UNDERWEIGHT" if drift_pp < 0 else "")
        drift_lines.append(
            f"  {ac:<22}  current={cur_pct:.1f}%  target={tgt_pct:.1f}%  "
            f"drift={drift_pp:+.1f}pp{flag}"
        )
    drift_table = "\n".join(drift_lines)

    # 2. Single-position exposure check — read limit from JSON (fallback 5.0)
    single_pos_limit = pf.get("single_position_limit_pct", 5.0)
    position_lines = [f"SINGLE-POSITION WEIGHTS vs {single_pos_limit:.0f}% IPS LIMIT:"]
    ips_breach_flags: list[str] = []
    for h in pf["holdings"]:
        weight_pct = h["current_mv"] / aum * 100
        breach = f" ← IPS BREACH (>{single_pos_limit:.0f}%)" if weight_pct > single_pos_limit else ""
        position_lines.append(
            f"  {h['ticker']:<14}  {h['name']:<42}  "
            f"{h['asset_class']:<22}  weight={weight_pct:.2f}%{breach}"
        )
        if breach:
            ips_breach_flags.append(f"{h['ticker']} ({weight_pct:.2f}%)")
    position_table = "\n".join(position_lines)

    # 3. FX floor check — read floor from JSON (fallback 60.0)
    fx_floor = pf.get("fx_floor_pct", 60.0)
    sek_mv = sum(h["current_mv"] for h in pf["holdings"] if h["fx_exposure"] == "SEK")
    sek_pct = sek_mv / aum * 100
    fx_status = (
        f"SEK base-currency share: {sek_pct:.1f}% "
        f"(policy floor = {fx_floor:.0f}%) "
        + ("— BREACH" if sek_pct < fx_floor else "— OK")
    )

    # 4. IPS summary
    ips_section = (
        "IPS LIMITS:\n"
        f"  - Single-position limit: ≤{single_pos_limit:.0f}% of AUM per holding\n"
        f"  - FX policy: {pf['fx_policy']}\n"
        f"  - {fx_status}\n"
        + (f"  - Breached positions: {', '.join(ips_breach_flags)}\n" if ips_breach_flags else "")
    )

    # 5. Intel signals
    intel: IntelFindings = fetch_intel()
    intel_lines = ["MARKET INTEL SIGNALS (source: world_monitor_snapshot):"]
    for finding in intel.items:
        intel_lines.append(
            f"  [{finding.live_or_snapshot}] {finding.source} | {finding.metric} = {finding.value} "
            f"| relevance: {finding.relevance}"
        )
    intel_section = "\n".join(intel_lines)

    # 6. Doc IDs to cite
    citation_note = (
        "SOURCE DOC IDs FOR evidence_refs:\n"
        "  - Portfolio data  → doc_id: bergstrom_portfolio_q1_2026\n"
        "  - IPS document    → doc_id: bergstrom_ips\n"
        "  - Market intel    → doc_id: world_monitor_snapshot\n"
    )

    return "\n\n".join([
        drift_table,
        position_table,
        ips_section,
        intel_section,
        citation_note,
        (
            "TASK: Based on the above, emit 2–3 OpportunitySignals. "
            "The US-tech overweight (+5.0 pp) and Gulf real estate overweight (+5.0 pp) "
            "MUST appear as drift signals. "
            "Beyond the drift signals, add a signal for any IPS violation you can see in the "
            "data (FX base-currency floor breach, single-position limit breach) and for any "
            "clear macro or event signal from the intel feed. "
            "Every signal must include at least one evidence_ref with the appropriate doc_id above."
        ),
    ])
