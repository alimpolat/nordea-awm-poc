# Nordea AWM AI POC — agentic advisor-flow demo

**A five-stage agentic system that drafts a private-banking Monday brief for a Nordic UHNW family office.** Built as a demonstrator for the Nordea Head-of-AWM-AI mandate — opportunity identification, client insights, meeting preparation, grounded recommendations, follow-up capture. Every next-best-action is cited; every uncertainty says *"I don't know."*

**Live demo:** https://alimpolat-nordea-awm-poc.hf.space

| | |
|---|---|
| **Runtime** | Hugging Face Spaces (Docker) · production target EU-region Cloud Run |
| **Models** | Vertex AI Gemini 2.5 Pro / Flash · gemini-embedding-001 |
| **Orchestration** | LlamaIndex Workflows · 5 stages · 4 parallel specialists |
| **Retrieval** | Qdrant + BM25 + RRF + Contextual Retrieval + Gemini Flash listwise reranker |
| **Eval** | Ragas 0.4.3 — recall@5 **0.909** · faithfulness **0.963** · answer-relevancy **1.000** |
| **Frontend** | React 18 + Vite + Tailwind + Apache ECharts |
| **Tracing** | Arize Phoenix |

> **POC status.** Proof-of-concept built for evaluation purposes. The Bergström family office and all associated data are entirely fictitious. The system is not connected to any live financial data or production Nordea systems.

## What to look at

Open the live URL and spend three minutes with:

1. **The Monday brief** — loads on page open from a pre-generated cache. Opportunity flags, next-best-actions (each cited to a source document), and risk panel. The Bergström portfolio is running +5pp over IPS target in US tech and Gulf real estate — those are the three NBAs.
2. **The advisor chat** — below the brief, ask a question about the portfolio or IPS. The RAG pipeline retrieves from the Bergström corpus (portfolio, IPS, three meeting notes, IMF/ECB/BIS reports). Ask "What is Bergström's allocation to Japanese equities?" — it answers "I don't know" correctly.
3. **The citations** — every recommendation card shows source doc IDs and the exact chunk used. This is what makes the system demonstrable to a regulated private bank rather than a demo toy.

## Run locally

**Prerequisites:** Python 3.12 + [uv](https://docs.astral.sh/uv/) + a GCP service-account JSON with Vertex AI permissions.

```bash
git clone https://github.com/alimpolat/nordea-awm-poc
cd nordea-awm-poc
uv sync --extra dev
cp .env.example .env             # fill in your own values
uv run --no-sync python scripts/smoke_vertex.py
uv run --no-sync uvicorn app.main:app --port 8001 --reload
# open http://localhost:8001
```

**Rebuild the brief and index:**

```bash
uv run --no-sync python -m app.orchestrator          # 5-stage agentic flow
uv run --no-sync python scripts/build_index.py       # Qdrant + BM25 index
```

**Run the offline test suite (67 tests, no live Vertex calls):**

```bash
uv run --no-sync pytest -m "not live" -q
```

## Design and evaluation docs

For the full styled write-ups, browse these in the repo:

- [`README.html`](README.html) — the styled version of this file
- [`ARCHITECTURE.html`](ARCHITECTURE.html) — system design, animated 5-stage flow diagram, retrieval pipeline, model routing table, decisions log
- [`EVAL.html`](EVAL.html) — evaluation trust receipt with per-intent breakdown, Phoenix trace screenshot, hand review
- [`docs/`](docs/) — research notes, full design specification, architecture flow doc

## Repository layout

<details>
<summary>Click to expand</summary>

```
nordea-awm-poc/
├── app/
│   ├── main.py               # FastAPI app, /api routes
│   ├── orchestrator.py       # LlamaIndex Workflows — 5-stage runner
│   ├── schemas.py            # Pydantic output schemas (BriefSchema, NBAs, …)
│   ├── agents/               # one file per stage-agent
│   ├── retrieval/            # hybrid RAG stack
│   ├── intel/                # World_Monitor client + snapshot fallback
│   └── llm/                  # Vertex AI client wrapper
├── frontend/                 # React 18 + Vite + Tailwind source
├── data/                     # raw corpus (fixtures, snapshots)
├── scripts/                  # build_index, fetch_corpus, smoke_vertex
├── eval/                     # Ragas runner, 35 synthetic Qs, results
├── tests/                    # pytest suite (67 offline tests)
├── fallback/                 # static brief snapshot (cache fallback)
├── docs/                     # design + architecture HTML docs
├── Dockerfile                # multi-stage Docker build
├── cloudbuild.yaml           # Cloud Run build config
└── pyproject.toml            # uv / pytest config
```

</details>

---

Built 2026-05-26 · Alim Polat · [linkedin.com/in/alimpolat](https://linkedin.com/in/alimpolat)
