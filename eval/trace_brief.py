"""Eval script — Phoenix tracing + brief generation.

Runs the full 5-stage agentic pipeline once with Phoenix-OSS tracing enabled,
so every Gemini call and every stage boundary produces an OpenInference span
visible in the Phoenix UI.

Usage
-----
    uv run --no-sync python eval/trace_brief.py

After the brief finishes (~2 min) the Phoenix process stays alive.
Open http://localhost:6006 to inspect the trace tree.

Span hierarchy produced
-----------------------
  [AGENT]  generate_brief
    [CHAIN]  stage_1_opportunity_scout       (parallel)
      [CHAIN]  stage/opportunity_scout
        [LLM]  gemini/<flash>
    [CHAIN]  stage_2_client_insights         (parallel)
      [CHAIN]  stage/client_insights
        [LLM]  gemini/<flash>
    [CHAIN]  stage_3a_planner
      [CHAIN]  stage/planner
        [LLM]  gemini/<pro>
    [CHAIN]  stage_3b_intel_gathering        (parallel)
      [CHAIN]  stage/intel_gathering
        [LLM]  gemini/<flash>
    [CHAIN]  stage_3c_macro                  (parallel)
      [CHAIN]  stage/macro
        [LLM]  gemini/<flash>
    [CHAIN]  stage_3d_portfolio              (parallel)
      [CHAIN]  stage/portfolio
        [LLM]  gemini/<flash>
    [CHAIN]  stage_3e_news                   (parallel)
      [CHAIN]  stage/news
        [LLM]  gemini/<flash>
    [CHAIN]  stage_4_synthesizer
      [CHAIN]  stage/synthesizer
        [LLM]  gemini/<pro>
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path (needed when running as a script)
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ---------------------------------------------------------------------------
# 1. Init tracing FIRST — patches generate() and run_agent_sync before any
#    agent module is imported by the orchestrator.
# ---------------------------------------------------------------------------
from app.observability import init_tracing  # noqa: E402

print("[trace_brief] Starting Phoenix UI + wiring OTel spans …")
init_tracing(launch_ui=True)

# Short pause to let Phoenix bind its HTTP port
time.sleep(3)

# ---------------------------------------------------------------------------
# 2. Import orchestrator AFTER patching (so the patched generate() is active)
# ---------------------------------------------------------------------------
from app.orchestrator import generate_brief, save_brief_cache, BRIEF_CACHE_PATH  # noqa: E402
from opentelemetry import trace  # noqa: E402
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes  # noqa: E402

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND

# ---------------------------------------------------------------------------
# 3. Instrumented brief runner — adds stage CHAIN spans + top-level AGENT span
# ---------------------------------------------------------------------------

# Map of agent names → stage labels for the CHAIN spans the orchestrator emits
# We instrument each stage via a wrapper around generate_brief.
# The per-agent CHAIN spans are already emitted by the patched run_agent_sync.
# We still want stage-level CHAIN spans to group parallel agents.

from app.orchestrator import (  # noqa: E402
    generate_brief as _orig_generate_brief,
)
import app.agents.opportunity_scout as _opp_mod  # noqa: E402
import app.agents.client_insights as _ci_mod  # noqa: E402
import app.agents.planner as _plan_mod  # noqa: E402
import app.agents.intel_gathering as _ig_mod  # noqa: E402
import app.agents.macro as _macro_mod  # noqa: E402
import app.agents.portfolio as _port_mod  # noqa: E402
import app.agents.news as _news_mod  # noqa: E402
import app.agents.synthesizer as _synth_mod  # noqa: E402


def _make_stage_wrapper(original_fn, stage_label: str):
    """Wrap an async agent run() with a CHAIN span named stage_label."""
    from app.observability import get_tracer

    async def _wrapped(*args, **kwargs):
        t = get_tracer()
        with t.start_as_current_span(stage_label) as span:
            span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
            return await original_fn(*args, **kwargs)

    return _wrapped


# Patch each agent's run() with a stage-level CHAIN span
_opp_mod.run = _make_stage_wrapper(_opp_mod.run, "stage_1_opportunity_scout")
_ci_mod.run = _make_stage_wrapper(_ci_mod.run, "stage_2_client_insights")
_plan_mod.run = _make_stage_wrapper(_plan_mod.run, "stage_3a_planner")
_ig_mod.run = _make_stage_wrapper(_ig_mod.run, "stage_3b_intel_gathering")
_macro_mod.run = _make_stage_wrapper(_macro_mod.run, "stage_3c_macro")
_port_mod.run = _make_stage_wrapper(_port_mod.run, "stage_3d_portfolio")
_news_mod.run = _make_stage_wrapper(_news_mod.run, "stage_3e_news")
_synth_mod.run = _make_stage_wrapper(_synth_mod.run, "stage_4_synthesizer")


async def _run_traced_brief() -> None:
    """Top-level coroutine — wraps generate_brief in an AGENT span."""
    from app.observability import get_tracer

    t = get_tracer()
    meeting_dt = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)

    with t.start_as_current_span("generate_brief") as span:
        span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        span.set_attribute("client_id", "bergstrom")
        span.set_attribute("meeting_datetime", meeting_dt.isoformat())

        print(f"[trace_brief] Running 5-stage pipeline for 'bergstrom' — {meeting_dt.isoformat()} …")
        t0 = time.perf_counter()

        brief = await generate_brief("bergstrom", meeting_dt)
        elapsed = time.perf_counter() - t0

        span.set_attribute("brief.intel_mode", brief.intel_mode)
        span.set_attribute("brief.nba_count", len(brief.three_nbas))
        span.set_attribute("brief.opportunity_count", len(brief.opportunities))
        span.set_attribute("brief.risk_flag_count", len(brief.risk_flags))
        span.set_attribute("elapsed_seconds", round(elapsed, 2))

    print(f"[trace_brief] Pipeline finished in {elapsed:.1f}s")
    print(f"  intel_mode      : {brief.intel_mode}")
    print(f"  three_nbas      : {len(brief.three_nbas)}")
    print(f"  opportunities   : {len(brief.opportunities)}")
    print(f"  risk_flags      : {len(brief.risk_flags)}")
    print(f"  weekend_changes : {len(brief.weekend_changes)}")

    # Write brief cache as a side-effect
    save_brief_cache(brief)
    print(f"[trace_brief] Brief cache → {BRIEF_CACHE_PATH}")

    # Give the span processor a moment to flush before we print the ready message
    time.sleep(2)
    print()
    print("=" * 60)
    print("  Phoenix UI at http://localhost:6006 — traces ready")
    print("=" * 60)
    print()
    print("[trace_brief] Phoenix process running. Press Ctrl+C to stop.")


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(_run_traced_brief())

    # Keep the Phoenix server alive so the screenshot can be taken
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[trace_brief] Shutting down.")
        import phoenix as px

        px.close_app()
