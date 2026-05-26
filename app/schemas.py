"""All Pydantic models for the POC, in one file (small, related, change together)."""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low_needs_verification"]


class EvidenceRef(BaseModel):
    doc_id: str
    chunk_id: str | None = None
    source_uri: str | None = None
    excerpt: str | None = None


class OpportunitySignal(BaseModel):
    trigger_type: Literal["drift", "macro", "event", "ips_violation"]
    asset_class: str
    magnitude: float
    confidence: Confidence
    suggested_topic: str
    evidence_refs: list[EvidenceRef]


class OpportunitySignals(BaseModel):
    items: list[OpportunitySignal]


class Holding(BaseModel):
    ticker: str
    name: str
    asset_class: str
    quantity: float
    cost_basis: float
    current_mv: float
    ytd_return_pct: float
    dividend_ytd: float
    fx_exposure: str


class ClientSnapshot(BaseModel):
    client_id: str
    client_name: str
    aum_sek: float
    holdings: list[Holding]
    target_allocation: dict[str, float]
    stated_concerns: list[str]
    restrictions: list[str]
    last_meeting_date: date


class IntelFinding(BaseModel):
    source: str
    metric: str
    value: str | float
    as_of: datetime
    relevance: str
    live_or_snapshot: Literal["live", "snapshot"]


class IntelFindings(BaseModel):
    items: list[IntelFinding]


class MacroFinding(BaseModel):
    claim: str
    evidence_chunks: list[EvidenceRef]
    confidence: Confidence
    impact_on_portfolio: str


class MacroFindings(BaseModel):
    items: list[MacroFinding]


class PortfolioFinding(BaseModel):
    drift_signals: list[dict]
    ips_compliance: list[dict]
    ytd_summary: dict
    opportunities: list[str]
    computation_trace: str | None = None


class NewsItem(BaseModel):
    headline: str
    source_uri: str
    ts: datetime
    relevance_tag: str


class NewsFindings(BaseModel):
    items: list[NewsItem]


class NextBestAction(BaseModel):
    title: str
    rationale: str
    projected_impact: str
    confidence: Confidence
    evidence_refs: list[EvidenceRef]
    computation_trace: str | None = None
    suggested_priority: Literal["primary", "secondary", "tertiary"]


class RiskFlag(BaseModel):
    kind: Literal["concentration", "fx", "regulatory", "liquidity", "none"]
    severity: Literal["info", "watch", "action", "none"]
    note: str


class BriefSchema(BaseModel):
    client_id: str
    generated_at: datetime
    intel_mode: Literal["live", "snapshot", "mixed"]
    opportunities: list[OpportunitySignal]
    weekend_changes: list[NewsItem | MacroFinding]
    three_nbas: list[NextBestAction] = Field(min_length=1, max_length=3)
    risk_flags: list[RiskFlag]


class ChatResponse(BaseModel):
    answer: str
    cited_refs: list[EvidenceRef]
    confidence: Confidence


class ChatRequest(BaseModel):
    client_id: str
    question: str
    history: list[dict] = []   # optional prior turns; unused by the minimal impl


class HitlRequest(BaseModel):
    client_id: str
    nba_title: str | None = None   # which NBA the action targets (by title)
    nba_index: int | None = None
    reason: str | None = None      # for reject/edit
    edited_text: str | None = None  # for edit


class Plan(BaseModel):
    specialists_to_invoke: list[Literal["intel", "macro", "portfolio", "news"]]
    sub_questions: dict[str, list[str]]
    output_schema_name: str = "BriefSchema"
