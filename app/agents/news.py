"""Stage 3e — News specialist.

Uses Google Search grounding (Vertex AI) to surface recent news relevant to
the Bergström portfolio themes (Gulf real estate, US tech, green bonds, Nordic
macro), then re-emits the findings as structured NewsFindings.

Design: two-pass approach (grounding + response_schema cannot be combined)
------------------------------------------------------------------------
Pass 1 — grounded fetch (Google Search tool, NO response_schema):
  Ask Gemini Flash for news relevant to the Bergström themes from the past
  ~2 weeks.  Capture the raw grounded text AND the real source URIs from
  `response.candidates[0].grounding_metadata.grounding_chunks`.

Pass 2 — structured re-emit (response_schema=NewsFindings, NO tools):
  Feed the grounded text + real URIs back to Flash and ask it to produce
  structured NewsFindings where every source_uri must come from the list
  provided (never invented).

Honesty / anti-hallucination:
  After pass 2 any NewsItem whose source_uri either does not start with
  "http" or cannot be matched (substring) to one of the real grounding URIs
  is dropped.  An empty result (NewsFindings(items=[])) is the correct,
  plan-sanctioned outcome when grounding is unavailable.

Graceful degradation:
  If pass 1 raises an exception or returns no grounding URIs, a WARNING is
  logged and an empty NewsFindings is returned.  The orchestrator/synthesizer
  handles the empty-news case.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

from google.genai import types

from app.llm.prompt_loader import load_prompt
from app.llm.vertex_client import generate
from app.schemas import NewsFindings, NewsItem
from app.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Soft timeout for pass-1 grounding fetch. Google Search grounding legitimately takes
# ~25-40s for a multi-theme query (measured ~28s for the 4 default themes, returning 25
# real source URIs). Because the brief is generated in the BACKGROUND and served from
# cache (Thomas never waits), there is no 60s request-path pressure here — so we allow a
# generous cap and only degrade to empty NewsFindings if grounding genuinely hangs well
# beyond the expected window. (An earlier 15s cap silently killed the ~28s call, which is
# why news looked "unavailable".)
_PASS1_TIMEOUT_S: float = 90.0

_DEFAULT_QUESTIONS: list[str] = [
    "What are the latest Gulf real-estate market developments relevant to GCC investors?",
    "What are recent US tech-sector valuation or earnings stories this week?",
    "Are there any new green-bond issuances or ESG finance stories relevant to EU investors?",
    "What is the latest Nordic macro news (Riksbank, Swedish/Norwegian economy)?",
]

# Relevance theme keywords → tag mapping (used in pass-2 prompt hint only)
_THEMES = (
    "Gulf real estate, GCC markets, Dubai property, Saudi real estate",
    "US large-cap technology valuations, tech earnings, FAANG, AI chip",
    "green bonds, ESG regulation, sustainable finance, climate bond",
    "Nordic macro, Riksbank, Norges Bank, Swedish economy, Norwegian economy",
)

_TAGS = (
    "gulf_real_estate",
    "us_tech_valuation",
    "green_bonds",
    "nordic_macro",
)

# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def run(sub_questions: list[str]) -> NewsFindings:
    """Async entry point — wraps the blocking two-pass pipeline in a thread pool."""
    return await asyncio.to_thread(_run_sync, sub_questions)


# ---------------------------------------------------------------------------
# Synchronous implementation
# ---------------------------------------------------------------------------


def _run_sync(sub_questions: list[str]) -> NewsFindings:
    questions = sub_questions if sub_questions else _DEFAULT_QUESTIONS[:]

    # Pass 1: grounded search — with a hard timeout to protect the orchestrator budget.
    # Google Search grounding can take 15-40s; we cap at _PASS1_TIMEOUT_S and degrade
    # gracefully to empty NewsFindings rather than blocking the parallel Stage-3 gather.
    # We use shutdown(wait=False, cancel_futures=True) to abandon the background thread
    # immediately on timeout so asyncio.to_thread can return promptly.
    try:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_pass1_grounded_fetch, questions)
        timed_out = False
        try:
            grounded_text, real_uris = future.result(timeout=_PASS1_TIMEOUT_S)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if timed_out:
            logger.warning(
                "News specialist: pass-1 grounded fetch timed out after %.0fs — "
                "returning empty NewsFindings.",
                _PASS1_TIMEOUT_S,
            )
            return NewsFindings(items=[])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "News specialist: pass-1 grounded fetch raised an exception — "
            "returning empty NewsFindings. Error: %s: %s",
            type(exc).__name__,
            exc,
        )
        return NewsFindings(items=[])

    if not real_uris:
        logger.warning(
            "News specialist: pass-1 returned no grounding URIs "
            "(Google Search grounding may not be enabled for this project). "
            "Returning empty NewsFindings."
        )
        return NewsFindings(items=[])

    logger.info(
        "News specialist: pass-1 grounding returned %d URI(s): %s",
        len(real_uris),
        [u for u, _t in real_uris],
    )

    # Pass 2: structured re-emit
    try:
        raw_findings = _pass2_structured_reemit(questions, grounded_text, real_uris)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "News specialist: pass-2 structured re-emit failed — "
            "returning empty NewsFindings. Error: %s: %s",
            type(exc).__name__,
            exc,
        )
        return NewsFindings(items=[])

    # Validate: drop any item whose source_uri is fabricated
    validated = _validate_uris(raw_findings, real_uris)

    logger.info(
        "News specialist: %d item(s) survived URI validation (raw: %d).",
        len(validated.items),
        len(raw_findings.items),
    )
    return validated


# ---------------------------------------------------------------------------
# Pass 1 — grounded fetch
# ---------------------------------------------------------------------------


def _build_search_prompt(questions: list[str]) -> str:
    """Build the search-oriented prompt for pass 1."""
    q_lines = ["=== NEWS SUB-QUESTIONS FROM PLANNER ==="]
    for i, q in enumerate(questions, start=1):
        q_lines.append(f"  [{i}] {q}")
    questions_section = "\n".join(q_lines)

    theme_lines = ["=== KEY PORTFOLIO THEMES TO COVER ==="]
    for tag, theme in zip(_TAGS, _THEMES):
        theme_lines.append(f"  [{tag}] {theme}")
    themes_section = "\n".join(theme_lines)

    task = (
        "TASK (Pass 1 — grounded search):\n"
        "Using Google Search grounding, find recent news stories (last ~2 weeks) that "
        "are directly relevant to the sub-questions and portfolio themes above. "
        "For each story provide: the exact headline or a close paraphrase, the "
        "publication date, and the source URL. "
        "Do NOT fabricate stories — only report what your grounded search actually returns. "
        "If grounding returns nothing for a theme, say so explicitly."
    )

    return "\n\n".join([questions_section, themes_section, task])


_PASS1_SYSTEM_INSTRUCTION = (
    "You are a financial news researcher with access to Google Search grounding. "
    "Your task is to find recent, real news stories relevant to the questions provided. "
    "Respond in plain text (NOT JSON). For each story you find, write the headline, "
    "the source domain, and a one-sentence summary. "
    "Only report stories that your grounded search actually returns — never fabricate."
)


def _pass1_grounded_fetch(
    questions: list[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Call Gemini Flash with Google Search grounding; return (grounded_text, real_uris).

    real_uris is a list of (uri, title) tuples extracted from grounding_metadata.
    Raises on any Vertex/network error (caller handles).

    IMPORTANT: Pass-1 uses a plain-text system instruction (not the JSON-output news prompt)
    because Vertex AI only populates grounding_chunks when the model returns natural-language
    text.  If the system prompt contains a JSON output contract the model emits structured
    JSON directly, which suppresses grounding_chunks entirely.
    """
    prompt = _build_search_prompt(questions)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    response = generate(
        model=settings.gemini_model_flash,
        contents=prompt,
        system_instruction=_PASS1_SYSTEM_INSTRUCTION,
        tools=[grounding_tool],
    )

    grounded_text = response.text or ""

    # Extract grounding URIs from metadata (guard every access)
    real_uris: list[tuple[str, str]] = []
    try:
        candidates = response.candidates or []
        if candidates:
            gm = candidates[0].grounding_metadata
            if gm is not None:
                chunks = gm.grounding_chunks or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web is not None:
                        uri = getattr(web, "uri", None) or ""
                        title = getattr(web, "title", None) or ""
                        if uri.startswith("http"):
                            real_uris.append((uri, title))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "News specialist: error extracting grounding URIs from metadata: %s",
            exc,
        )

    return grounded_text, real_uris


# ---------------------------------------------------------------------------
# Pass 2 — structured re-emit
# ---------------------------------------------------------------------------


def _build_reemit_prompt(
    questions: list[str],
    grounded_text: str,
    real_uris: list[tuple[str, str]],
) -> str:
    """Build the pass-2 prompt that asks the model to produce structured NewsFindings."""
    q_lines = ["=== ORIGINAL NEWS SUB-QUESTIONS ==="]
    for i, q in enumerate(questions, start=1):
        q_lines.append(f"  [{i}] {q}")
    questions_section = "\n".join(q_lines)

    grounded_section = (
        "=== GROUNDED SEARCH RESULTS (Pass 1 output) ===\n"
        + (grounded_text or "(empty)")
    )

    uri_lines = [
        f"=== REAL SOURCE URIs FROM GOOGLE SEARCH GROUNDING ({len(real_uris)} found) ===",
        "You MUST only use source_uris from this list. Do NOT invent URLs.",
    ]
    for idx, (uri, title) in enumerate(real_uris, start=1):
        uri_lines.append(f"  [{idx}] uri={uri!r}  title={title!r}")
    uris_section = "\n".join(uri_lines)

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    tag_list = ", ".join(_TAGS) + ", other"

    task = (
        "TASK (Pass 2 — structured re-emit):\n"
        "Using ONLY the grounded search results and real source URIs listed above, "
        "produce a NewsFindings object. Rules:\n"
        f"  - source_uri: MUST be taken verbatim from the 'REAL SOURCE URIs' list above.\n"
        "  - headline: the actual headline text or a direct paraphrase (not a summary sentence).\n"
        f"  - ts: publication datetime in ISO 8601 format. If unknown use '{now_iso}'.\n"
        f"  - relevance_tag: one of [{tag_list}] matching the story's theme.\n"
        "  - Aim for 3-6 items. Omit marginal or off-topic stories.\n"
        "  - If the grounded search returned nothing useful, return items=[].\n"
        "  - NEVER fabricate a URL. If you cannot assign a real URI from the list, omit the item."
    )

    return "\n\n".join([questions_section, grounded_section, uris_section, task])


def _pass2_structured_reemit(
    questions: list[str],
    grounded_text: str,
    real_uris: list[tuple[str, str]],
) -> NewsFindings:
    """Call Gemini Flash with response_schema=NewsFindings; return parsed NewsFindings.

    No grounding tool in this call — only structured output.
    Raises RuntimeError if the model returns no parsed output.
    """
    prompt = _build_reemit_prompt(questions, grounded_text, real_uris)

    response = generate(
        model=settings.gemini_model_flash,
        contents=prompt,
        system_instruction=load_prompt("news"),
        response_schema=NewsFindings,
    )

    if response.parsed is None:
        raise RuntimeError(
            f"news: pass-2 model returned no structured output. Raw: {response.text!r}"
        )

    return response.parsed


# ---------------------------------------------------------------------------
# URI validation
# ---------------------------------------------------------------------------


def _uri_is_real(candidate_uri: str, real_uris: list[tuple[str, str]]) -> bool:
    """Return True if candidate_uri matches one of the real grounding URIs.

    Matching: exact OR substring (the model may shorten/reshape the URL slightly,
    e.g. stripping query params).  We accept if either string is contained in
    the other, normalised to lowercase.
    """
    if not candidate_uri.startswith("http"):
        return False
    candidate_lower = candidate_uri.lower()
    for uri, _title in real_uris:
        uri_lower = uri.lower()
        if candidate_lower == uri_lower:
            return True
        if candidate_lower in uri_lower or uri_lower in candidate_lower:
            return True
    return False


def _validate_uris(
    findings: NewsFindings,
    real_uris: list[tuple[str, str]],
) -> NewsFindings:
    """Drop any NewsItem whose source_uri is not in the real grounding URI list."""
    validated: list[NewsItem] = []
    for item in findings.items:
        if _uri_is_real(item.source_uri, real_uris):
            validated.append(item)
        else:
            logger.warning(
                "News specialist: dropping item with fabricated/unverified URI %r "
                "(headline=%r)",
                item.source_uri,
                item.headline,
            )
    return NewsFindings(items=validated)
