"""Stage 3b — Intel Gathering specialist.

Receives sub-questions from the Planner, fetches ground-truth signals from
World Monitor (snapshot or live), asks Gemini Flash to SELECT and ANNOTATE
the relevant signals, then RECONCILES the model output against the ground-truth
to prevent fabrication.

Design decisions
----------------
* Ground-truth signals come from ``fetch_intel()`` — no model can invent them.
* The model's only creative role is writing ``relevance`` text tailored to the
  sub-questions. All other fields (source, metric, value, as_of,
  live_or_snapshot) are OVERWRITTEN from the ground-truth after the call.
* Reconciliation matches on ``metric`` (case-insensitive substring / equality).
  Items the model emits that cannot be matched to a real signal are silently
  dropped — the POC never presents fabricated market data.
* If reconciliation leaves zero items (e.g. the model hallucinated every metric
  name), we fall back to returning ``fetch_intel()`` unchanged so the caller
  always gets usable output.
"""
import asyncio

from app.agents._base import run_agent_sync
from app.intel.world_monitor_client import fetch_intel
from app.schemas import IntelFinding, IntelFindings

# ---------------------------------------------------------------------------
# Default sub-questions (used when the caller passes an empty list)
# ---------------------------------------------------------------------------

_DEFAULT_QUESTIONS: list[str] = [
    "What current market signals are most relevant to the Bergström family office's holdings?",
]


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def run(sub_questions: list[str]) -> IntelFindings:
    """Async entry point — wraps the blocking Gemini call in a thread pool."""
    return await asyncio.to_thread(_run_sync, sub_questions)


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------


def _run_sync(sub_questions: list[str]) -> IntelFindings:
    questions = sub_questions if sub_questions else _DEFAULT_QUESTIONS[:]

    # 1. Ground-truth signals from World Monitor (snapshot or live)
    ground_truth: IntelFindings = fetch_intel()

    # 2. Ask Flash to select and annotate relevant signals
    contents = _build_contents(questions, ground_truth)
    model_output: IntelFindings = run_agent_sync("intel_gathering", contents, IntelFindings)

    # 3. Reconcile model output against ground-truth
    reconciled = _reconcile(model_output, ground_truth)

    # 4. Fall back to raw ground-truth if reconciliation leaves nothing
    if not reconciled.items:
        return ground_truth

    return reconciled


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _reconcile(model_output: IntelFindings, ground_truth: IntelFindings) -> IntelFindings:
    """Overwrite all factual fields from ground-truth; keep only model's relevance text.

    Matching strategy: for each model item, find a ground-truth signal whose
    ``metric`` either equals or is a case-insensitive substring match of the
    model item's ``metric``.  First match wins.  Unmatched model items are
    dropped.

    The overwritten fields are: source, metric, value, as_of, live_or_snapshot.
    Only ``relevance`` is kept from the model.
    """
    # Build a lookup: normalised metric → ground-truth IntelFinding
    gt_by_metric: dict[str, IntelFinding] = {}
    for gt_item in ground_truth.items:
        gt_by_metric[gt_item.metric.lower()] = gt_item

    kept: list[IntelFinding] = []
    seen_gt_metrics: set[str] = set()  # prevent duplicate reconciliations

    for model_item in model_output.items:
        model_metric_lower = model_item.metric.lower()
        matched_gt: IntelFinding | None = None

        # Exact match first
        if model_metric_lower in gt_by_metric:
            matched_gt = gt_by_metric[model_metric_lower]
        else:
            # Substring match: model metric in gt metric or gt metric in model metric
            for gt_metric_lower, gt_item in gt_by_metric.items():
                if model_metric_lower in gt_metric_lower or gt_metric_lower in model_metric_lower:
                    matched_gt = gt_item
                    break

        if matched_gt is None:
            # No matching ground-truth signal — drop the model item
            continue

        gt_key = matched_gt.metric.lower()
        if gt_key in seen_gt_metrics:
            # Already used this ground-truth signal — skip duplicate
            continue
        seen_gt_metrics.add(gt_key)

        # Build reconciled item: ground-truth facts + model relevance text
        reconciled_item = IntelFinding(
            source=matched_gt.source,
            metric=matched_gt.metric,
            value=matched_gt.value,
            as_of=matched_gt.as_of,
            relevance=model_item.relevance,          # Only model contribution
            live_or_snapshot=matched_gt.live_or_snapshot,
        )
        kept.append(reconciled_item)

    return IntelFindings(items=kept)


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _build_contents(questions: list[str], ground_truth: IntelFindings) -> str:
    """Build the user-turn string for the Intel Gathering LLM call.

    Sections:
      1. Sub-questions from the Planner.
      2. Ground-truth signals — metric, value, as_of, live_or_snapshot, relevance
         (giving the model the existing relevance as context to build on).
      3. Task instruction.
    """
    # 1. Sub-questions
    q_lines = ["=== PLANNER SUB-QUESTIONS (intel specialist) ==="]
    for i, q in enumerate(questions, start=1):
        q_lines.append(f"  [{i}] {q}")
    questions_section = "\n".join(q_lines)

    # 2. Ground-truth signals
    signal_lines = [
        "=== WORLD MONITOR SIGNALS (ground-truth; do NOT fabricate or alter these values) ==="
    ]
    for item in ground_truth.items:
        signal_lines.append(
            f"  metric={item.metric!r}  "
            f"value={item.value}  "
            f"as_of={item.as_of.isoformat()}  "
            f"source={item.source!r}  "
            f"live_or_snapshot={item.live_or_snapshot!r}  "
            f"current_relevance={item.relevance!r}"
        )
    signals_section = "\n".join(signal_lines)

    # 3. Task instruction
    task = (
        "TASK:\n"
        "From the ground-truth signals above, SELECT only those that are relevant to "
        "the Planner's sub-questions. For each selected signal, write a new 1-2 sentence "
        "relevance field that ties the signal directly to the sub-questions and the "
        "Bergström family office's portfolio.\n\n"
        "IMPORTANT CONSTRAINTS:\n"
        "  - You MUST use only the metric names and values from the ground-truth list above.\n"
        "  - Do NOT invent new signals or alter metric names, values, sources, or "
        "    live_or_snapshot tags.\n"
        "  - live_or_snapshot: preserve EXACTLY as shown — never relabel snapshot as live.\n"
        "  - If a signal is not relevant to any sub-question, omit it.\n"
        "  - Return at least 1 and at most all signals (do not artificially limit selection).\n"
        "  - Use the metric name exactly as shown (copy-paste) so reconciliation works.\n"
    )

    return "\n\n".join([questions_section, signals_section, task])
