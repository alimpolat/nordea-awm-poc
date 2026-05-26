"""Diagnostic probe: is Google Search grounding actually engaging on this Vertex project?

The News specialist (Stage 3e) got non-empty text but zero grounding_chunks. This probe
inspects the FULL grounding metadata to tell apart the possible causes:
  - model never searched      -> web_search_queries empty/None (config/prompt issue)
  - searched but no sources    -> queries present, grounding_chunks empty (metadata/shape)
  - tool rejected / restricted -> exception (entitlement/region issue)
Run: uv run --no-sync python scripts/probe_grounding.py
"""
from google.genai import types

from app.llm.vertex_client import client
from app.settings import settings


def _inspect(label: str, model: str, query: str, tool: types.Tool) -> None:
    print(f"\n===== {label} · model={model} =====")
    print(f"query: {query!r}")
    try:
        cfg = types.GenerateContentConfig(tools=[tool])
        r = client.models.generate_content(model=model, contents=query, config=cfg)
    except Exception as e:  # noqa: BLE001
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return
    text = (r.text or "")
    print(f"  text[:200]: {text[:200]!r}")
    cands = r.candidates or []
    if not cands:
        print("  NO candidates")
        return
    c = cands[0]
    print(f"  finish_reason: {getattr(c, 'finish_reason', None)}")
    gm = getattr(c, "grounding_metadata", None)
    print(f"  grounding_metadata present: {gm is not None}")
    if gm is not None:
        wsq = getattr(gm, "web_search_queries", None)
        print(f"  web_search_queries: {wsq}")
        gc = getattr(gm, "grounding_chunks", None) or []
        print(f"  grounding_chunks: {len(gc)}")
        for ch in gc[:6]:
            w = getattr(ch, "web", None)
            print(f"    - uri={getattr(w, 'uri', None)!r} title={getattr(w, 'title', None)!r}")
        sep = getattr(gm, "search_entry_point", None)
        print(f"  search_entry_point present: {sep is not None}")
        rm = getattr(gm, "retrieval_metadata", None)
        print(f"  retrieval_metadata: {rm}")


def main() -> None:
    q_current = "What are the latest news headlines about ECB interest rate decisions this week?"
    q_force = (
        "Use Google Search to find recent (last 2 weeks) news about Dubai / Gulf "
        "commercial real estate and Saudi REITs. Cite the sources."
    )
    flash = settings.gemini_model_flash
    pro = settings.gemini_model_pro

    # 1) modern tool constructor (Gemini 2.x on Vertex)
    _inspect("google_search (Flash)", flash, q_current, types.Tool(google_search=types.GoogleSearch()))
    _inspect("google_search forced (Flash)", flash, q_force, types.Tool(google_search=types.GoogleSearch()))
    # 2) same on Pro (in case grounding is model-gated)
    _inspect("google_search (Pro)", pro, q_current, types.Tool(google_search=types.GoogleSearch()))
    # 3) legacy retrieval tool (Gemini 1.5 era) — may error on 2.x, that's informative
    try:
        legacy = types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
        _inspect("google_search_retrieval (Flash, legacy)", flash, q_current, legacy)
    except Exception as e:  # noqa: BLE001
        print(f"\n  legacy tool construction failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
