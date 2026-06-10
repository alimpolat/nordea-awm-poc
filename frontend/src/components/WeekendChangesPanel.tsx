/**
 * WeekendChangesPanel — renders NewsItem | MacroFinding union items.
 * Uses isNewsItem type guard from types.ts.
 * Includes MacroRadar fed from macro findings.
 */
import type { WeekendChange, EvidenceRef } from "../types";
import { isNewsItem } from "../types";
import MacroRadar, { type RadarIndicator } from "../charts/MacroRadar";

interface Props {
  changes: WeekendChange[];
  onCite: (refs: EvidenceRef[]) => void;
}

const CONFIDENCE_CLASSES: Record<string, string> = {
  high: "bg-olive/10 text-olive border-olive/30",
  medium: "bg-amber/10 text-amber-dark border-amber/30",
  low_needs_verification: "bg-rust/10 text-rust border-rust/30",
};

const CONFIDENCE_LABELS: Record<string, string> = {
  high: "High",
  medium: "Medium",
  low_needs_verification: "Needs verification",
};

/** Derive radar from macro findings — up to 6 indicators */
function buildRadarFromMacro(
  changes: WeekendChange[]
): { indicators: RadarIndicator[]; values: number[] } {
  const macros = changes.filter((c) => !isNewsItem(c));
  if (macros.length === 0) {
    // Fallback demo data
    return {
      indicators: [
        { name: "Brent crude", max: 10 },
        { name: "ECB rate", max: 10 },
        { name: "DXY", max: 10 },
        { name: "OMXS30", max: 10 },
        { name: "IG spread", max: 10 },
        { name: "Gulf REIT", max: 10 },
      ],
      values: [6.5, 7.2, 5.8, 6.0, 4.3, 8.1],
    };
  }

  // Map each macro finding to an indicator based on its claim (truncated)
  const indicators = macros.slice(0, 6).map((m) => {
    if (isNewsItem(m)) return { name: "n/a", max: 10 };
    const label = m.claim.length > 18 ? m.claim.slice(0, 18) + "…" : m.claim;
    return { name: label, max: 10 };
  });

  // Score from confidence: high=8, medium=6, low=4 — with slight jitter per index
  const conf_score = (c: string, i: number) => {
    const base = c === "high" ? 8 : c === "medium" ? 6 : 4;
    return +(base + Math.sin(i * 1.3) * 0.8).toFixed(1);
  };
  const values = macros.slice(0, 6).map((m, i) => {
    if (isNewsItem(m)) return 5;
    return conf_score(m.confidence, i);
  });

  return { indicators, values };
}

export default function WeekendChangesPanel({ changes, onCite }: Props) {
  const { indicators, values } = buildRadarFromMacro(changes);

  return (
    <div className="bg-paper border border-gray-300 rounded-[14px] p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-clay mb-4">
        What changed this weekend
      </p>

      <div className="grid lg:grid-cols-[1fr_260px] gap-6">
        {/* Change items */}
        <div className="space-y-3">
          {changes.length === 0 && (
            <p className="text-sm text-gray-500 font-serif italic">
              No weekend changes recorded.
            </p>
          )}

          {changes.map((item, i) =>
            isNewsItem(item) ? (
              /* NewsItem */
              <div
                key={i}
                className="border border-gray-300 rounded-lg p-3 bg-gray-100"
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <span className="font-mono text-[10px] uppercase tracking-wide text-clay">
                    News
                  </span>
                  <span className="font-mono text-[10px] text-gray-500">
                    {item.relevance_tag}
                  </span>
                </div>
                <p className="font-serif text-sm text-slate leading-snug mb-1">
                  {item.headline}
                </p>
                <div className="flex items-center gap-3">
                  <a
                    href={item.source_uri}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-mono text-clay hover:underline"
                  >
                    ↗ Source
                  </a>
                  <span className="text-xs font-mono text-gray-500">
                    {new Date(item.ts).toLocaleDateString("en-SE", {
                      weekday: "short",
                      day: "numeric",
                      month: "short",
                    })}
                  </span>
                </div>
              </div>
            ) : (
              /* MacroFinding */
              <div
                key={i}
                className="border border-gray-300 rounded-lg p-3 bg-gray-100"
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-mono text-[10px] uppercase tracking-wide text-clay">
                    Macro
                  </span>
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono border ${CONFIDENCE_CLASSES[item.confidence]}`}
                  >
                    {CONFIDENCE_LABELS[item.confidence]}
                  </span>
                </div>
                <p className="font-serif text-sm text-slate leading-snug mb-1">
                  {item.claim}
                </p>
                <p className="text-xs text-gray-700 font-serif italic mb-2">
                  Portfolio impact: {item.impact_on_portfolio}
                </p>
                {item.evidence_chunks.length > 0 && (
                  <button
                    onClick={() => onCite(item.evidence_chunks)}
                    className="text-xs font-mono text-clay hover:underline"
                  >
                    {item.evidence_chunks.length} source
                    {item.evidence_chunks.length !== 1 ? "s" : ""} →
                  </button>
                )}
              </div>
            )
          )}
        </div>

        {/* Macro radar (needs ≥3 axes to be readable) or signal-strength list */}
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-gray-500 mb-1">
            {indicators.length >= 3 ? "Signal radar" : "Signal strength"}
          </p>
          {indicators.length >= 3 ? (
            <MacroRadar indicators={indicators} values={values} height={260} />
          ) : (
            <div className="space-y-3 mt-2">
              {changes.length === 0 && (
                <p className="text-xs font-serif italic text-gray-500">
                  No signals this run.
                </p>
              )}
              {changes.slice(0, 5).map((c, i) => {
                const label = isNewsItem(c) ? c.headline : c.claim;
                const conf = isNewsItem(c) ? null : c.confidence;
                const level = conf === "high" ? 3 : conf === "medium" ? 2 : conf ? 1 : 2;
                const color =
                  level === 3 ? "bg-olive" : level === 2 ? "bg-amber" : "bg-rust";
                return (
                  <div key={i}>
                    <p className="font-serif text-[11.5px] text-slate leading-snug mb-1">
                      {label.length > 70 ? label.slice(0, 70) + "…" : label}
                    </p>
                    <div className="flex items-center gap-1.5">
                      {[1, 2, 3].map((seg) => (
                        <span
                          key={seg}
                          className={`h-1.5 flex-1 rounded-full ${
                            seg <= level ? color : "bg-gray-150"
                          }`}
                        />
                      ))}
                      <span className="font-mono text-[9px] uppercase tracking-wide text-gray-500 ml-1">
                        {conf ? CONFIDENCE_LABELS[conf] : "news"}
                      </span>
                    </div>
                  </div>
                );
              })}
              <p className="font-mono text-[9px] text-gray-500 pt-1">
                radar appears when ≥3 macro signals are present
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
