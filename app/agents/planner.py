"""Stage 3a — Planner agent.

Receives the OpportunitySignals (Stage 1) and ClientSnapshot (Stage 2), then
authors a Plan: 2-4 focused sub-questions per specialist grounded in the
client's specific concerns and signals.

Design decisions
----------------
* The POC always invokes all four specialists. The LLM is told this explicitly in
  the system prompt (planner.txt) and again in the user turn.
* A Python-level invariant enforces it regardless of model output: after the LLM
  call we unconditionally set specialists_to_invoke = ["intel","macro","portfolio",
  "news"] and fill in any missing sub_question keys with sensible defaults so the
  orchestrator's asyncio.gather over the four specialists always has inputs.
* Async interface (asyncio.to_thread) mirrors opportunity_scout and client_insights
  so the orchestrator can gather Stage-1/2/3a in parallel.
"""
import asyncio
from datetime import datetime

from app.agents._base import run_agent_sync
from app.schemas import ClientSnapshot, OpportunitySignals, Plan
from app.settings import settings

# ---------------------------------------------------------------------------
# Fallback sub-questions if the model omits a specialist
# ---------------------------------------------------------------------------
_FALLBACK_QUESTIONS: dict[str, list[str]] = {
    "intel": [
        "What current market signals are most relevant to this client's holdings?"
    ],
    "macro": [
        "What are the key macro developments this week that affect the client's portfolio?"
    ],
    "portfolio": [
        "How do the current asset-class weights compare to the IPS target bands?"
    ],
    "news": [
        "What major news items from the past 7 days are relevant to the client's holdings?"
    ],
}

_ALL_SPECIALISTS = ["intel", "macro", "portfolio", "news"]


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def run(
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    meeting_datetime: datetime,
) -> Plan:
    """Async entry point — wraps the blocking Gemini Pro call in a thread pool."""
    return await asyncio.to_thread(
        _run_sync, opportunity_signals, client_snapshot, meeting_datetime
    )


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------

def _run_sync(
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    meeting_datetime: datetime,
) -> Plan:
    contents = _build_contents(opportunity_signals, client_snapshot, meeting_datetime)
    # Pro provides deepest reasoning for sub-question authoring — the Planner is the
    # design-spec "deepest reasoning" stage that seeds all four specialists.
    # Brief generation runs in the background (cache-first serving), so latency
    # here is not on the user's critical path.
    plan: Plan = run_agent_sync(
        "planner",
        contents,
        Plan,
        model=settings.gemini_model_pro,
    )

    # -----------------------------------------------------------------------
    # POC invariant: all four specialists are always invoked.
    # The model may omit one or return a subset — enforce it here.
    # -----------------------------------------------------------------------
    plan.specialists_to_invoke = _ALL_SPECIALISTS[:]

    for specialist in _ALL_SPECIALISTS:
        existing = plan.sub_questions.get(specialist, [])
        if not any(q.strip() for q in existing):
            plan.sub_questions[specialist] = _FALLBACK_QUESTIONS[specialist][:]
        else:
            # Strip blank entries that slipped through
            plan.sub_questions[specialist] = [q for q in existing if q.strip()]

    if not plan.output_schema_name:
        plan.output_schema_name = "BriefSchema"

    return plan


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _build_contents(
    opportunity_signals: OpportunitySignals,
    client_snapshot: ClientSnapshot,
    meeting_datetime: datetime,
) -> str:
    """Build the user-turn string for the Planner LLM call.

    Includes:
      1. Meeting datetime for temporal context.
      2. Each opportunity signal (trigger_type, asset_class, magnitude, suggested_topic).
      3. The client's stated_concerns and restrictions from the snapshot.
      4. An explicit instruction to author 2-4 sub-questions per specialist, always
         invoking all four, grounded in the Gulf concentration, US-tech valuation,
         green-bond reallocation, and FX-floor breach concerns.
    """
    # 1. Meeting context — use %d (zero-padded) for Windows portability
    meeting_str = meeting_datetime.strftime("%A %d %B %Y at %H:%M")
    meeting_section = f"MEETING DATETIME: {meeting_str}"

    # 2. Opportunity signals
    signal_lines = ["OPPORTUNITY SIGNALS FROM STAGE 1:"]
    for i, sig in enumerate(opportunity_signals.items, start=1):
        signal_lines.append(
            f"  [{i}] trigger_type={sig.trigger_type}  "
            f"asset_class={sig.asset_class!r}  "
            f"magnitude={sig.magnitude:+.1f}pp  "
            f"confidence={sig.confidence}  "
            f"suggested_topic={sig.suggested_topic!r}"
        )
    signals_section = "\n".join(signal_lines)

    # 3. Client concerns and restrictions
    concerns_lines = ["CLIENT STATED CONCERNS (from Stage 2 — most recent meeting):"]
    for c in client_snapshot.stated_concerns:
        concerns_lines.append(f"  - {c}")
    concerns_section = "\n".join(concerns_lines)

    restrictions_lines = ["CLIENT IPS RESTRICTIONS:"]
    for r in client_snapshot.restrictions:
        restrictions_lines.append(f"  - {r}")
    restrictions_section = "\n".join(restrictions_lines)

    # 4. Client portfolio summary (for context)
    portfolio_lines = [
        f"CLIENT: {client_snapshot.client_name} ({client_snapshot.client_id})",
        f"AUM: SEK {client_snapshot.aum_sek:,.0f}",
        "TARGET ALLOCATION:",
    ]
    for ac, weight in client_snapshot.target_allocation.items():
        portfolio_lines.append(f"  {ac}: {weight * 100:.1f}%")
    portfolio_section = "\n".join(portfolio_lines)

    # 5. Task instruction
    task = (
        "TASK:\n"
        "Author a Plan for the Monday advisor brief. You MUST invoke all four "
        "specialists: intel, macro, portfolio, news. For each specialist, write "
        "2-4 focused sub-questions grounded in the client's specific concerns "
        "above.\n\n"
        "Key themes to address across the four specialists:\n"
        "  - Gulf concentration (high allocation, regulatory/market risk)\n"
        "  - US-tech valuation discomfort (overweight, stretch valuations)\n"
        "  - Green-bond reallocation interest (IPS-compliant ESG options)\n"
        "  - FX-floor breach risk (SEK base-currency minimum)\n\n"
        "Generic questions are not acceptable. Ground every question in the "
        "client's actual holdings, concerns, and signals above.\n\n"
        "Always set output_schema_name to 'BriefSchema'."
    )

    return "\n\n".join([
        meeting_section,
        signals_section,
        concerns_section,
        restrictions_section,
        portfolio_section,
        task,
    ])
