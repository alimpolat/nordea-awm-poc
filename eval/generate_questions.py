"""Task 5.1 — Generate synthetic question set for Ragas evaluation.

Generates 30 corpus-grounded questions (5 per intent bucket × 6 buckets) via
Gemini 2.5 Pro structured output, then appends 5 hand-authored hard cases.
Output: eval/synthetic_qs.jsonl (35 lines, one JSON object per line).

Usage:
    uv run --no-sync python eval/generate_questions.py
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pypdf
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from repo root or eval/ directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.fixtures import load_ips_text, load_meeting_notes, load_portfolio  # noqa: E402
from app.llm.vertex_client import generate  # noqa: E402
from app.settings import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_DOC_IDS = [
    "bergstrom_portfolio_q1_2026",
    "bergstrom_ips",
    "bergstrom_meeting_notes_2026-01-14",
    "bergstrom_meeting_notes_2026-02-18",
    "bergstrom_meeting_notes_2026-04-14",
    "bis_quarterly_2025_q4",
    "bis_quarterly_2026_q1",
    "ecb_economic_bulletin_2026",
    "imf_weo_2026_04",
]

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
OUTPUT_PATH = Path(__file__).resolve().parent / "synthetic_qs.jsonl"

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class QItem(BaseModel):
    question: str
    expected_doc_ids: list[str]
    intent: str


class QSet(BaseModel):
    items: list[QItem]


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def _load_pdf_pages(pdf_name: str, max_pages: int = 12) -> str:
    """Extract text from the first max_pages of a corpus PDF."""
    path = CORPUS_DIR / pdf_name
    reader = pypdf.PdfReader(str(path))
    pages = reader.pages[:max_pages]
    return " ".join(
        " ".join(p.extract_text(extraction_mode="plain").split())
        for p in pages
        if p.extract_text()
    )


def load_corpus_texts() -> dict[str, str]:
    """Load all corpus documents as plain text, keyed by doc_id."""
    portfolio = load_portfolio("bergstrom")
    texts: dict[str, str] = {}

    # Portfolio (JSON → pretty repr, truncated to ~4 KB for the prompt)
    port_str = json.dumps(portfolio, indent=2, ensure_ascii=False)
    texts["bergstrom_portfolio_q1_2026"] = port_str[:6000]

    # IPS
    texts["bergstrom_ips"] = load_ips_text("bergstrom")

    # Meeting notes (3 files, sorted chronologically)
    note_dates = ["2026-01-14", "2026-02-18", "2026-04-14"]
    notes = load_meeting_notes("bergstrom")
    for date, note in zip(note_dates, notes):
        texts[f"bergstrom_meeting_notes_{date}"] = note[:3000]

    # Macro PDFs — first 12 pages each
    for pdf_name, doc_id in [
        ("bis_quarterly_2025_q4.pdf", "bis_quarterly_2025_q4"),
        ("bis_quarterly_2026_q1.pdf", "bis_quarterly_2026_q1"),
        ("ecb_economic_bulletin_2026.pdf", "ecb_economic_bulletin_2026"),
        ("imf_weo_2026_04.pdf", "imf_weo_2026_04"),
    ]:
        try:
            texts[doc_id] = _load_pdf_pages(pdf_name)[:5000]
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not load {pdf_name}: {exc}")
            texts[doc_id] = ""

    return texts


# ---------------------------------------------------------------------------
# Bucket definitions and prompt templates
# ---------------------------------------------------------------------------

BUCKET_DEFS = {
    "lookup": (
        "SINGLE-FACT RETRIEVAL questions. Each question asks for one specific fact "
        "that can be answered from a single document with minimal inference. "
        "Example: 'What is Bergström's target allocation to Gulf real estate?'"
    ),
    "multi_hop": (
        "MULTI-HOP questions that require combining information from TWO OR MORE "
        "different documents. Each question must cite concerns or data that appear "
        "in different docs (e.g. a meeting-note concern + the IPS policy limit). "
        "Example: 'At the April 2026 meeting the family expressed discomfort with "
        "US-tech multiples — what is the IPS upper band for that sleeve?'"
    ),
    "quantitative": (
        "QUANTITATIVE / CALCULATION questions grounded in the portfolio numbers. "
        "Each question requires reading a specific number and either reporting it "
        "or doing a simple comparison/arithmetic. "
        "Example: 'How many percentage points is the US-tech sleeve over its target "
        "as of Q1 2026?'"
    ),
    "contextual": (
        "CONTEXTUAL questions that need surrounding context to interpret. The answer "
        "depends on understanding the situation around a fact — not just the raw fact. "
        "Example: 'Given the IMF's global growth outlook, what is the implication for "
        "Bergström's European fixed-income sleeve?'"
    ),
    "macro_reasoning": (
        "MACRO REASONING questions sourced primarily from the IMF WEO, ECB bulletin, "
        "or BIS quarterly docs. Each question asks about a macro event, rate decision, "
        "or outlook mentioned in those documents. "
        "Example: 'What did the ECB indicate about its deposit facility rate in early 2026?'"
    ),
    "nba_justification": (
        "NEXT-BEST-ACTION JUSTIFICATION questions asking WHY a particular recommendation "
        "is warranted, grounded in the portfolio + IPS + meeting notes. "
        "Example: 'Why should the US-tech sleeve be trimmed, given the IPS and the "
        "family's stated concerns?'"
    ),
}


def build_prompt(bucket: str, bucket_desc: str, corpus_excerpt: str) -> str:
    valid_ids_str = "\n".join(f"  - {d}" for d in VALID_DOC_IDS)
    return textwrap.dedent(f"""
    You are a financial QA dataset generator for a Retrieval-Augmented Generation eval.

    TASK: Generate exactly 5 questions for the bucket "{bucket}".
    Bucket definition: {bucket_desc}

    CORPUS EXCERPT (use ONLY the information below to ground your questions):
    {corpus_excerpt}

    VALID DOC IDS (expected_doc_ids MUST be drawn exclusively from this list):
    {valid_ids_str}

    RULES:
    1. Each question must be answerable from the corpus excerpt.
    2. Every entry in expected_doc_ids must be from the valid doc_id list above.
    3. For multi_hop questions include ≥2 doc_ids; for others ≥1.
    4. Questions must be specific and unambiguous — no generic finance questions.
    5. intent for all items must be exactly: "{bucket}"
    6. Return exactly 5 items.
    """).strip()


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate_bucket(
    bucket: str,
    desc: str,
    corpus_texts: dict[str, str],
) -> list[dict]:
    """Call Gemini 2.5 Pro to generate 5 questions for one bucket."""
    # Build a focused excerpt: client docs + macro docs, relevant to the bucket
    if bucket in ("macro_reasoning", "contextual"):
        # Macro-heavy
        relevant_docs = [
            "imf_weo_2026_04",
            "ecb_economic_bulletin_2026",
            "bis_quarterly_2026_q1",
            "bis_quarterly_2025_q4",
            "bergstrom_portfolio_q1_2026",
            "bergstrom_ips",
        ]
    elif bucket == "lookup":
        # Client docs are sufficient
        relevant_docs = [
            "bergstrom_portfolio_q1_2026",
            "bergstrom_ips",
            "bergstrom_meeting_notes_2026-01-14",
            "bergstrom_meeting_notes_2026-02-18",
            "bergstrom_meeting_notes_2026-04-14",
        ]
    elif bucket in ("multi_hop", "nba_justification"):
        # Needs both client + some macro
        relevant_docs = [
            "bergstrom_portfolio_q1_2026",
            "bergstrom_ips",
            "bergstrom_meeting_notes_2026-02-18",
            "bergstrom_meeting_notes_2026-04-14",
            "ecb_economic_bulletin_2026",
            "bis_quarterly_2026_q1",
        ]
    else:
        # quantitative — primarily portfolio + IPS
        relevant_docs = [
            "bergstrom_portfolio_q1_2026",
            "bergstrom_ips",
        ]

    # Build excerpt (keep total prompt under ~8 KB)
    excerpt_parts = []
    budget = 7000
    for doc_id in relevant_docs:
        text = corpus_texts.get(doc_id, "")
        if not text:
            continue
        snippet = text[:min(len(text), budget // len(relevant_docs))]
        excerpt_parts.append(f"[DOC: {doc_id}]\n{snippet}")
        if sum(len(p) for p in excerpt_parts) > budget:
            break
    corpus_excerpt = "\n\n".join(excerpt_parts)

    prompt = build_prompt(bucket, desc, corpus_excerpt)

    print(f"  Calling Gemini Pro for bucket '{bucket}'...")
    response = generate(
        model=settings.gemini_model_pro,
        contents=prompt,
        response_schema=QSet,
    )
    q_set: QSet = response.parsed

    results = []
    for item in q_set.items:
        # Sanitize: keep only valid doc_ids
        valid_ids = [d for d in item.expected_doc_ids if d in VALID_DOC_IDS]
        results.append(
            {
                "question": item.question,
                "expected_doc_ids": valid_ids,
                "intent": bucket,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Hand-authored hard cases (5)
# ---------------------------------------------------------------------------

HARD_CASES: list[dict] = [
    # (a) Named-entity disambiguation — Jabal Omar vs generic Gulf RE
    {
        "question": (
            "What is the market value and YTD return of Jabal Omar Development "
            "(Makkah) specifically, as distinct from the Emirates REIT (Dubai) position?"
        ),
        "expected_doc_ids": ["bergstrom_portfolio_q1_2026"],
        "intent": "hard_named_entity_disambiguation",
    },
    # (b) Multi-doc synthesis — macro + portfolio
    {
        "question": (
            "Given the ECB's monetary policy stance described in the 2026 bulletin "
            "and Bergström's current European fixed-income weight, is the portfolio "
            "over- or under-positioned versus its IPS target, and by how many "
            "percentage points?"
        ),
        "expected_doc_ids": [
            "ecb_economic_bulletin_2026",
            "bergstrom_portfolio_q1_2026",
            "bergstrom_ips",
        ],
        "intent": "hard_multi_doc_synthesis",
    },
    # (c) "I don't know" — corpus genuinely cannot answer
    {
        "question": "What is Bergström's allocation to Japanese equities?",
        "expected_doc_ids": [],
        "intent": "hard_unanswerable",
    },
    # (d) IPS-violation detection — SEK FX-floor breach
    {
        "question": (
            "Does Bergström's current portfolio satisfy the IPS FX policy requiring "
            "at least 60% SEK base-currency exposure? What is the actual figure and "
            "by how many percentage points does it deviate from the floor?"
        ),
        "expected_doc_ids": [
            "bergstrom_portfolio_q1_2026",
            "bergstrom_ips",
        ],
        "intent": "hard_ips_violation_detection",
    },
    # (e) Currency math — SEK base-currency share calculation
    {
        "question": (
            "Summing the SEK-denominated holdings and the Nordic equity sleeve, "
            "what fraction of the SEK 480M portfolio is held in instruments "
            "denominated in SEK on a direct (non-look-through) basis?"
        ),
        "expected_doc_ids": ["bergstrom_portfolio_q1_2026"],
        "intent": "hard_currency_math",
    },
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(records: list[dict]) -> None:
    """Raise AssertionError if any validation rule is violated."""
    assert len(records) == 35, f"Expected 35 records, got {len(records)}"
    for i, rec in enumerate(records):
        assert "question" in rec and rec["question"], f"Row {i}: missing question"
        assert "intent" in rec and rec["intent"], f"Row {i}: missing intent"
        assert "expected_doc_ids" in rec, f"Row {i}: missing expected_doc_ids"
        # Validate doc_ids (empty list is allowed for hard_unanswerable)
        for doc_id in rec["expected_doc_ids"]:
            assert doc_id in VALID_DOC_IDS, (
                f"Row {i}: invalid doc_id '{doc_id}' not in VALID_DOC_IDS"
            )
    print("Validation PASSED: 35 records, all doc_ids valid.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading corpus texts...")
    corpus_texts = load_corpus_texts()
    print(f"  Loaded {len(corpus_texts)} documents.")

    all_records: list[dict] = []
    bucket_counts: dict[str, int] = {}

    for bucket, desc in BUCKET_DEFS.items():
        records = generate_bucket(bucket, desc, corpus_texts)
        all_records.extend(records)
        bucket_counts[bucket] = len(records)
        print(f"  [{bucket}] -> {len(records)} questions generated")

    # Append hard cases
    all_records.extend(HARD_CASES)
    bucket_counts["hard_cases"] = len(HARD_CASES)

    # Validate
    print("\nValidating...")
    validate(all_records)

    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records)} records to {OUTPUT_PATH}")

    # Per-bucket summary
    print("\nPer-bucket counts:")
    for bucket, count in bucket_counts.items():
        print(f"  {bucket:35s}: {count}")

    # Sample 3 questions
    samples = [all_records[0], all_records[7], all_records[20]]
    print("\nSample questions:")
    for s in samples:
        print(f"\n  [{s['intent']}]")
        print(f"  Q: {s['question']}")
        print(f"  Doc IDs: {s['expected_doc_ids']}")


if __name__ == "__main__":
    main()
