/**
 * RiskFlags — risk flag list with severity-coloured badges.
 * Shows an explicit "No flags" (olive) when the array is empty.
 */
import type { RiskFlag } from "../types";

interface Props {
  flags: RiskFlag[];
}

const SEVERITY_CLASSES: Record<string, string> = {
  info: "bg-gray-150 text-gray-700 border-gray-300",
  watch: "bg-amber/10 text-amber-dark border-amber/40",
  action: "bg-rust/10 text-rust border-rust/40",
  none: "bg-gray-150 text-gray-500 border-gray-300",
};

const KIND_LABELS: Record<string, string> = {
  concentration: "CONCENTRATION",
  fx: "FX EXPOSURE",
  regulatory: "REGULATORY",
  liquidity: "LIQUIDITY",
  none: "NONE",
};

export default function RiskFlags({ flags }: Props) {
  return (
    <div className="bg-paper border border-gray-300 rounded-[14px] p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-clay mb-4">
        Risk Flags
      </p>

      {flags.length === 0 ? (
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono uppercase tracking-wide bg-olive/10 text-olive border border-olive/30">
            No flags
          </span>
          <span className="text-sm text-gray-500 font-serif">
            Portfolio is within all risk thresholds.
          </span>
        </div>
      ) : (
        <ul className="space-y-3">
          {flags.map((flag, i) => (
            <li key={i} className="flex items-start gap-3">
              <div className="flex gap-2 flex-shrink-0 pt-0.5">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wide border ${SEVERITY_CLASSES[flag.severity] ?? SEVERITY_CLASSES.info}`}
                >
                  {flag.severity.toUpperCase()}
                </span>
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wide bg-gray-100 text-gray-700 border border-gray-300">
                  {KIND_LABELS[flag.kind] ?? flag.kind.toUpperCase()}
                </span>
              </div>
              <p className="text-sm text-gray-700 font-serif leading-snug">{flag.note}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
