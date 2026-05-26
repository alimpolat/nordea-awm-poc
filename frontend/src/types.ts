/**
 * TypeScript interfaces mirroring app/schemas.py exactly.
 * Keep in sync with Pydantic models — the API returns these shapes.
 */

export type Confidence = "high" | "medium" | "low_needs_verification";

export interface EvidenceRef {
  doc_id: string;
  chunk_id?: string | null;
  source_uri?: string | null;
  excerpt?: string | null;
}

export interface OpportunitySignal {
  trigger_type: "drift" | "macro" | "event" | "ips_violation";
  asset_class: string;
  magnitude: number;
  confidence: Confidence;
  suggested_topic: string;
  evidence_refs: EvidenceRef[];
}

export interface Holding {
  ticker: string;
  name: string;
  asset_class: string;
  quantity: number;
  cost_basis: number;
  current_mv: number;
  ytd_return_pct: number;
  dividend_ytd: number;
  fx_exposure: string;
}

export interface ClientSnapshot {
  client_id: string;
  client_name: string;
  aum_sek: number;
  holdings: Holding[];
  target_allocation: Record<string, number>;
  stated_concerns: string[];
  restrictions: string[];
  last_meeting_date: string; // ISO date string
}

export interface IntelFinding {
  source: string;
  metric: string;
  value: string | number;
  as_of: string; // ISO datetime string
  relevance: string;
  live_or_snapshot: "live" | "snapshot";
}

export interface MacroFinding {
  claim: string;
  evidence_chunks: EvidenceRef[];
  confidence: Confidence;
  impact_on_portfolio: string;
}

export interface NewsItem {
  headline: string;
  source_uri: string;
  ts: string; // ISO datetime string
  relevance_tag: string;
}

export interface NextBestAction {
  title: string;
  rationale: string;
  projected_impact: string;
  confidence: Confidence;
  evidence_refs: EvidenceRef[];
  computation_trace?: string | null;
  suggested_priority: "primary" | "secondary" | "tertiary";
}

export interface RiskFlag {
  kind: "concentration" | "fx" | "regulatory" | "liquidity" | "none";
  severity: "info" | "watch" | "action" | "none";
  note: string;
}

/** BriefSchema.weekend_changes is a union of NewsItem | MacroFinding */
export type WeekendChange = NewsItem | MacroFinding;

export interface BriefSchema {
  client_id: string;
  generated_at: string; // ISO datetime string
  intel_mode: "live" | "snapshot" | "mixed";
  opportunities: OpportunitySignal[];
  weekend_changes: WeekendChange[];
  three_nbas: NextBestAction[];
  risk_flags: RiskFlag[];
}

export interface ChatResponse {
  answer: string;
  cited_refs: EvidenceRef[];
  confidence: Confidence;
}

export interface ChatRequest {
  client_id: string;
  question: string;
  history?: { role: string; content: string }[];
}

export interface HitlRequest {
  client_id: string;
  nba_title?: string | null;
  nba_index?: number | null;
  reason?: string | null;
  edited_text?: string | null;
}

/** Discriminator helper: narrows WeekendChange to NewsItem */
export const isNewsItem = (w: WeekendChange): w is NewsItem =>
  "source_uri" in w;
