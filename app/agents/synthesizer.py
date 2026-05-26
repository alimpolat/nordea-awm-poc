"""Stage 4 — Synthesizer agent.

Assembles the Monday brief (BriefSchema) from the plan, opportunity signals,
client snapshot, and the four Stage-3 specialists' findings (intel, macro,
portfolio, news).

Design decisions
----------------
* The model (Gemini Pro) synthesises judgment — which risks are most material,
  what three NBAs the advisor should lead with, how weekend macro/news changes
  should be framed.
* Python enforces deterministic / contractual fields AFTER the model responds:
  - client_id and generated_at are always overwritten with authoritative values.
  - intel_mode is derived from the live_or_snapshot tags in IntelFindings.
  - opportunities are passed through verbatim from Stage 1 (not regenerated).
  - Every NBA must have ≥1 evidence_ref (fallback appended if the model forgot).
  - weekend_changes falls back to macro + news items if the model returned empty.
  - risk_flags: if the portfolio shows an FX breach and the model emitted no flags
    at all, a Python-derived FX action flag is appended.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from app.agents._base import run_agent_sync
from app.schemas import (
    BriefSchema,
    ClientSnapshot,
    EvidenceRef,
    IntelFindings,
    MacroFinding,
    MacroFindings,
    NewsFindings,
    NewsItem,
    NextBestAction,
    OpportunitySignals,
    Plan,
    PortfolioFinding,
    RiskFlag,
)
from app.settings import settings


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def run(
    plan: Plan,
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    intel: IntelFindings,
    macro: MacroFindings,
    portfolio: PortfolioFinding,
    news: NewsFindings,
) -> BriefSchema:
    """Async entry-point — wraps the blocking pipeline in a thread pool."""
    return await asyncio.to_thread(
        _run_sync,
        plan,
        opportunity_signals,
        client_snapshot,
        intel,
        macro,
        portfolio,
        news,
    )


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------


def _run_sync(
    plan: Plan,
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    intel: IntelFindings,
    macro: MacroFindings,
    portfolio: PortfolioFinding,
    news: NewsFindings,
) -> BriefSchema:
    # 1. Build model input
    contents = _build_contents(
        plan, opportunity_signals, client_snapshot, intel, macro, portfolio, news
    )

    # 2. LLM call — Gemini Pro synthesises judgment.
    # The Synthesizer is the brief's quality-critical final stage per the design spec;
    # Pro's superior reasoning capacity ensures accurate risk framing and NBA ranking.
    # Brief generation runs in the background (cache-first serving), so latency
    # here is not on the user's critical path.
    brief: BriefSchema = run_agent_sync(
        "synthesizer",
        contents,
        BriefSchema,
        model=settings.gemini_model_pro,
    )

    # 3. Python reconciliation / enforcement (deterministic)
    brief = _reconcile(brief, opportunity_signals, client_snapshot, intel, macro, portfolio, news)

    return brief


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _build_contents(
    plan: Plan,
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    intel: IntelFindings,
    macro: MacroFindings,
    portfolio: PortfolioFinding,
    news: NewsFindings,
) -> str:
    """Build the user-turn string summarising ALL stage outputs for Gemini Pro."""
    sections: list[str] = []

    # --- Plan ---
    plan_lines = [
        "=== PLAN (Stage 3a) ===",
        f"Specialists invoked: {', '.join(plan.specialists_to_invoke)}",
    ]
    for specialist, questions in plan.sub_questions.items():
        plan_lines.append(f"  [{specialist}] " + " | ".join(questions))
    sections.append("\n".join(plan_lines))

    # --- Client snapshot ---
    snap_lines = [
        "=== CLIENT SNAPSHOT (Stage 2) ===",
        f"client_id:     {client_snapshot.client_id}",
        f"client_name:   {client_snapshot.client_name}",
        f"aum_sek:       SEK {client_snapshot.aum_sek:,.0f}",
        f"last_meeting:  {client_snapshot.last_meeting_date}",
        "stated_concerns:",
    ]
    for c in client_snapshot.stated_concerns:
        snap_lines.append(f"  - {c}")
    snap_lines.append("restrictions:")
    for r in client_snapshot.restrictions:
        snap_lines.append(f"  - {r}")
    snap_lines.append(
        f"target_allocation: "
        + ", ".join(f"{k}={v*100:.1f}%" if v < 1 else f"{k}={v:.1f}%" for k, v in client_snapshot.target_allocation.items())
    )
    sections.append("\n".join(snap_lines))

    # --- Opportunity signals (Stage 1) ---
    opp_lines = [
        f"=== OPPORTUNITY SIGNALS — Stage 1 ({len(opportunity_signals.items)} signals) ===",
        "These are passed through verbatim to brief.opportunities — do NOT alter them.",
    ]
    for sig in opportunity_signals.items:
        opp_lines.append(
            f"  trigger={sig.trigger_type}  asset_class={sig.asset_class!r}  "
            f"magnitude={sig.magnitude:.1f}pp  confidence={sig.confidence}  "
            f"topic={sig.suggested_topic!r}  "
            f"refs=[{', '.join(r.doc_id for r in sig.evidence_refs)}]"
        )
    sections.append("\n".join(opp_lines))

    # --- Intel findings (Stage 3b) ---
    intel_lines = [
        f"=== INTEL FINDINGS — Stage 3b ({len(intel.items)} findings) ===",
    ]
    _modes = {f.live_or_snapshot for f in intel.items}
    if not _modes or _modes == {"snapshot"}:
        _hint_mode = "snapshot"
    elif _modes == {"live"}:
        _hint_mode = "live"
    else:
        _hint_mode = "mixed"
    intel_lines.append(f"intel_mode: {_hint_mode}")
    for f in intel.items:
        intel_lines.append(
            f"  [{f.live_or_snapshot}] {f.source} | {f.metric} = {f.value} "
            f"(as_of={f.as_of.isoformat()}) | relevance: {f.relevance}"
        )
    sections.append("\n".join(intel_lines))

    # --- Macro findings (Stage 3c) ---
    macro_lines = [
        f"=== MACRO FINDINGS — Stage 3c ({len(macro.items)} findings) ===",
        "Citable doc_ids from these findings (use in evidence_refs for macro-linked NBAs):",
    ]
    macro_doc_ids: list[str] = []
    for mf in macro.items:
        cited = [r.doc_id for r in mf.evidence_chunks]
        macro_doc_ids.extend(cited)
        macro_lines.append(
            f"  claim: {mf.claim}"
        )
        macro_lines.append(
            f"    confidence={mf.confidence}  impact: {mf.impact_on_portfolio}"
        )
        if cited:
            macro_lines.append(f"    doc_ids: {', '.join(cited)}")
    sections.append("\n".join(macro_lines))

    # --- Portfolio finding (Stage 3d) ---
    port_lines = [
        "=== PORTFOLIO FINDING — Stage 3d (Python-computed, authoritative) ===",
    ]
    port_lines.append("drift_signals:")
    for d in portfolio.drift_signals:
        direction = (
            "OVERWEIGHT" if d.get("drift_pp", 0) > 0
            else ("UNDERWEIGHT" if d.get("drift_pp", 0) < 0 else "on-target")
        )
        port_lines.append(
            f"  {d.get('asset_class', '?'):<22}  "
            f"current={d.get('current_pct', 0):.1f}%  "
            f"target={d.get('target_pct', 0):.1f}%  "
            f"drift={d.get('drift_pp', 0):+.2f}pp  [{direction}]"
        )
    port_lines.append("ips_compliance:")
    for entry in portfolio.ips_compliance:
        if entry.get("rule") == "fx_floor_sek":
            port_lines.append(
                f"  FX floor: SEK={entry.get('sek_pct', '?'):.1f}%  "
                f"floor={entry.get('floor_pct', '?'):.0f}%  "
                f"status={entry.get('status', '?').upper()}"
            )
        elif entry.get("status") == "breach":
            port_lines.append(
                f"  Single-position breach: {entry.get('holding', entry.get('ticker', '?'))} "
                f"({entry.get('weight_pct', '?'):.2f}% > {entry.get('limit_pct', 5.0):.0f}%)"
            )
        else:
            port_lines.append(f"  {entry.get('rule', '?')}: {entry.get('detail', entry.get('status', '?'))}")
    port_lines.append(f"ytd_summary: {portfolio.ytd_summary}")
    port_lines.append("opportunities (from portfolio specialist):")
    for opp in portfolio.opportunities:
        port_lines.append(f"  - {opp}")
    if portfolio.computation_trace:
        port_lines.append(f"computation_trace:\n{portfolio.computation_trace}")
    sections.append("\n".join(port_lines))

    # --- News findings (Stage 3e) ---
    if news.items:
        news_lines = [f"=== NEWS FINDINGS — Stage 3e ({len(news.items)} items) ==="]
        for item in news.items:
            news_lines.append(
                f"  [{item.ts.date()}] {item.headline}  "
                f"(source={item.source_uri}  tag={item.relevance_tag})"
            )
        sections.append("\n".join(news_lines))
    else:
        sections.append(
            "=== NEWS FINDINGS — Stage 3e ===\n"
            "No live news available (Google Search grounding unavailable). "
            "Base weekend_changes on the MacroFindings above."
        )

    # --- Citable source IDs ---
    citable_ids = sorted(
        {"bergstrom_portfolio_q1_2026", "bergstrom_ips", "world_monitor_snapshot"}
        | set(macro_doc_ids)
    )
    cite_lines = [
        "=== CITABLE SOURCE IDs FOR evidence_refs ===",
        "Use ONLY these doc_ids in evidence_refs (do NOT invent IDs):",
    ]
    for did in citable_ids:
        cite_lines.append(f"  - {did}")
    sections.append("\n".join(cite_lines))

    # --- Task instruction ---
    sections.append(
        "=== TASK ===\n"
        "Produce a BriefSchema for this client's Monday brief:\n"
        "  1. three_nbas: 1–3 NextBestAction items ordered primary → secondary → tertiary.\n"
        "     Each NBA MUST have ≥1 evidence_ref from the citable source IDs list.\n"
        "     Lead with the most material issue (FX floor breach, overweights, macro signal).\n"
        "  2. weekend_changes: list the most relevant recent developments (news or macro).\n"
        "     If news is empty, draw from the MacroFindings.\n"
        "  3. risk_flags: surface concentration, fx, liquidity, or regulatory flags.\n"
        "     Every RiskFlag must have a non-null severity.\n"
        "  4. intel_mode: derive from IntelFindings live_or_snapshot tags.\n"
        "  5. opportunities: carry through the Stage 1 signals exactly — do NOT alter them."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Python reconciliation / enforcement
# ---------------------------------------------------------------------------


def _reconcile(
    brief: BriefSchema,
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    intel: IntelFindings,
    macro: MacroFindings,
    portfolio: PortfolioFinding,
    news: NewsFindings,
) -> BriefSchema:
    """Apply all deterministic contract rules after the model call."""

    # 3a. Overwrite identity / timestamp fields
    brief.client_id = client_snapshot.client_id
    brief.generated_at = datetime.now(timezone.utc)  # canonical UTC

    # 3b. Derive intel_mode from actual finding tags.
    #     Empty set (no intel items) → "snapshot" to avoid a misleading "mixed".
    modes = {f.live_or_snapshot for f in intel.items}
    if not modes or modes == {"snapshot"}:
        brief.intel_mode = "snapshot"
    elif modes == {"live"}:
        brief.intel_mode = "live"
    else:
        brief.intel_mode = "mixed"

    # 3c. Pass Stage 1 opportunities through verbatim.
    #     Use copies to prevent the cached brief from sharing mutable references
    #     with the upstream OpportunitySignals object.
    brief.opportunities = [o.model_copy() for o in opportunity_signals.items]

    # 3d. Enforce NBA evidence_refs and computation_trace
    fallback_ref = EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")

    drift_nba_keywords = {"drift", "ips", "rebalance", "trim", "overweight", "underweight", "fx"}

    enforced_nbas: list[NextBestAction] = []
    for nba in brief.three_nbas[:3]:  # defensive max-3
        # Evidence ref enforcement
        if not nba.evidence_refs:
            nba.evidence_refs = [fallback_ref]

        # computation_trace enforcement for drift/IPS related NBAs
        if nba.computation_trace is None:
            title_lower = nba.title.lower()
            rationale_lower = nba.rationale.lower()
            is_drift_related = any(
                kw in title_lower or kw in rationale_lower
                for kw in drift_nba_keywords
            )
            if is_drift_related and portfolio.computation_trace:
                nba.computation_trace = portfolio.computation_trace

        enforced_nbas.append(nba)

    # Ensure at least 1 NBA — if model returned none, synthesise from largest drift
    if not enforced_nbas:
        enforced_nbas = [_synthesise_fallback_nba(portfolio)]

    brief.three_nbas = enforced_nbas

    # 3e. weekend_changes fallback.
    #     Use copies to prevent the cached brief from sharing mutable references
    #     with the upstream News/Macro stage outputs.
    if not brief.weekend_changes:
        brief.weekend_changes = (
            [m.model_copy() for m in news.items]
            + [m.model_copy() for m in macro.items]
        )

    # 3f. risk_flags: ensure severity is always set; append FX flag if portfolio
    #     shows breach and model produced no flags at all
    _ensure_risk_flag_severities(brief)

    if not brief.risk_flags:
        fx_flag = _derive_fx_flag_from_portfolio(portfolio)
        if fx_flag:
            brief.risk_flags = [fx_flag]

    return brief


def _synthesise_fallback_nba(portfolio: PortfolioFinding) -> NextBestAction:
    """Create a minimal primary NBA from the largest portfolio drift signal.

    Belt-and-suspenders guard against direct/future callers that bypass the
    model path. Pydantic enforces ``min_length=1`` on ``three_nbas`` at parse
    time, so the reconciler only reaches this when the model returned an
    empty list after structural parsing succeeded.
    """
    if portfolio.drift_signals:
        largest = max(portfolio.drift_signals, key=lambda d: abs(d.get("drift_pp", 0)))
        ac = largest.get("asset_class", "unknown")
        drift = largest.get("drift_pp", 0)
        direction = "Trim" if drift > 0 else "Add to"
        title = f"{direction} {ac} position"
        rationale = (
            f"Portfolio {ac} is {abs(drift):.1f}pp {'over' if drift > 0 else 'under'} IPS target. "
            "Rebalancing restores mandate compliance."
        )
    else:
        title = "Review portfolio allocation"
        rationale = "Portfolio analysis identified compliance items for review."

    return NextBestAction(
        title=title,
        rationale=rationale,
        projected_impact="Restores IPS compliance and reduces mandate drift.",
        confidence="medium",
        evidence_refs=[EvidenceRef(doc_id="bergstrom_portfolio_q1_2026")],
        computation_trace=portfolio.computation_trace,
        suggested_priority="primary",
    )


def _ensure_risk_flag_severities(brief: BriefSchema) -> None:
    """Ensure every RiskFlag has a non-null severity (in-place).

    Belt-and-suspenders guard — Pydantic enforces valid severities at parse
    time, so this only fires on direct/future construction paths that bypass
    the model call (e.g. _derive_fx_flag_from_portfolio, unit tests).
    """
    valid_severities = {"info", "watch", "action", "none"}
    for flag in brief.risk_flags:
        if flag.severity not in valid_severities:
            flag.severity = "info"


def _derive_fx_flag_from_portfolio(portfolio: PortfolioFinding) -> RiskFlag | None:
    """Return an FX action flag if the portfolio shows an FX floor breach; else None."""
    for entry in portfolio.ips_compliance:
        if entry.get("rule") == "fx_floor_sek" and entry.get("status") == "breach":
            sek_pct = entry.get("sek_pct", "?")
            floor_pct = entry.get("floor_pct", 60)
            sek_pct_str = f"{sek_pct:.1f}%" if isinstance(sek_pct, float) else str(sek_pct)
            floor_pct_str = f"{floor_pct:.0f}%" if isinstance(floor_pct, (int, float)) else str(floor_pct)
            return RiskFlag(
                kind="fx",
                severity="action",
                note=(
                    f"SEK base share ({sek_pct_str}) is below the {floor_pct_str} IPS FX floor "
                    "— requires rebalancing into SEK-denominated instruments."
                ),
            )
    return None
