/**
 * OpportunitiesPanel — renders OpportunitySignal list with confidence badges.
 * Includes DriftBars fed from drift-type signals.
 * Clicking a signal's evidence calls onCite.
 */
import type { OpportunitySignal, EvidenceRef } from "../types";
import DriftBars, { type DriftEntry } from "../charts/DriftBars";

interface Props {
  opportunities: OpportunitySignal[];
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

const TRIGGER_CLASSES: Record<string, string> = {
  drift: "bg-clay/10 text-clay border-clay/30",
  macro: "bg-slate/10 text-slate border-slate/30",
  event: "bg-amber/10 text-amber-dark border-amber/30",
  ips_violation: "bg-rust/10 text-rust border-rust/30",
};

const TRIGGER_LABELS: Record<string, string> = {
  drift: "DRIFT",
  macro: "MACRO",
  event: "EVENT",
  ips_violation: "IPS VIOLATION",
};

export default function OpportunitiesPanel({ opportunities, onCite }: Props) {
  // Extract drift signals for DriftBars
  const driftEntries: DriftEntry[] = opportunities
    .filter((o) => o.trigger_type === "drift")
    .map((o) => ({
      asset_class: o.asset_class,
      drift_pp: o.magnitude,
    }));

  return (
    <div className="bg-paper border border-gray-300 rounded-[14px] p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-clay mb-4">
        Opportunities
      </p>

      <div className="grid lg:grid-cols-[1fr_300px] gap-6">
        {/* Signal list */}
        <div className="space-y-3">
          {opportunities.length === 0 && (
            <p className="text-sm text-gray-500 font-serif italic">
              No opportunities flagged this week.
            </p>
          )}

          {opportunities.map((signal, i) => (
            <div
              key={i}
              className="border border-gray-300 rounded-lg p-3 bg-gray-100"
            >
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono border ${TRIGGER_CLASSES[signal.trigger_type] ?? "bg-gray-150 text-gray-700 border-gray-300"}`}
                >
                  {TRIGGER_LABELS[signal.trigger_type] ?? signal.trigger_type.toUpperCase()}
                </span>
                <span className="font-mono text-[11px] text-gray-500">
                  {signal.asset_class}
                </span>
                <span
                  className={`ml-auto inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono border ${CONFIDENCE_CLASSES[signal.confidence]}`}
                >
                  {CONFIDENCE_LABELS[signal.confidence]}
                </span>
              </div>

              <p className="font-serif text-sm text-slate leading-snug mb-1">
                {signal.suggested_topic}
              </p>

              <div className="flex items-center gap-3 mt-1">
                <span className="font-mono text-[11px] text-gray-500">
                  Magnitude:{" "}
                  <span className="font-semibold text-slate">
                    {signal.magnitude > 0 ? "+" : ""}
                    {signal.magnitude.toFixed(1)} pp
                  </span>
                </span>

                {signal.evidence_refs.length > 0 && (
                  <button
                    onClick={() => onCite(signal.evidence_refs)}
                    className="text-xs font-mono text-clay hover:underline"
                  >
                    {signal.evidence_refs.length} evidence →
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Drift chart — only if there are drift signals */}
        {driftEntries.length > 0 && (
          <div>
            <p className="font-mono text-[10px] uppercase tracking-widest text-gray-500 mb-1">
              Allocation drift
            </p>
            <DriftBars drift={driftEntries} height={220} />
          </div>
        )}
      </div>
    </div>
  );
}
