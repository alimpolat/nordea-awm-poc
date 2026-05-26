/**
 * NbaCard — the HITL centerpiece.
 * Approve / Edit / Regenerate / Reject wired to postHitl.
 * HITL state is local (useState).
 * Includes NbaProjection chart with demo-derived series.
 */
import { useState } from "react";
import type { NextBestAction, EvidenceRef } from "../types";
import { postHitl } from "../api";
import NbaProjection from "../charts/NbaProjection";

interface Props {
  nba: NextBestAction;
  index: number;
  clientId: string;
  onCite: (refs: EvidenceRef[]) => void;
}

type HitlStatus =
  | "idle"
  | "approved"
  | "edited"
  | "rejected"
  | "regenerating";

const PRIORITY_CLASSES: Record<string, string> = {
  primary: "bg-clay/10 text-clay border-clay/40",
  secondary: "bg-olive/10 text-olive border-olive/40",
  tertiary: "bg-amber/10 text-amber-dark border-amber/40",
};

const CONFIDENCE_CLASSES: Record<string, string> = {
  high: "bg-olive/10 text-olive border-olive/30",
  medium: "bg-amber/10 text-amber-dark border-amber/30",
  low_needs_verification: "bg-rust/10 text-rust border-rust/30",
};

const CONFIDENCE_LABELS: Record<string, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low_needs_verification: "Needs verification",
};

/** Generate a deterministic 12-month projection from base + impact string */
function buildProjectionSeries(impactStr: string) {
  const BASE = 480_000_000;
  // Extract first number from projected_impact string as % lift (fallback 0.8)
  const match = impactStr.match(/([0-9]+(?:\.[0-9]+)?)/);
  const liftPct = match ? parseFloat(match[1]) / 100 : 0.008;

  const withoutAction = Array.from({ length: 12 }, (_, i) =>
    Math.round(BASE * (1 + 0.003 * i))
  );
  const withAction = Array.from({ length: 12 }, (_, i) =>
    Math.round(BASE * (1 + 0.003 * i + liftPct * i * (1 - Math.exp(-i / 4))))
  );
  return { withAction, withoutAction };
}

export default function NbaCard({ nba, index, clientId, onCite }: Props) {
  const [status, setStatus] = useState<HitlStatus>("idle");
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState(nba.rationale);
  const [rejectReason, setRejectReason] = useState("");
  const [showRejectInput, setShowRejectInput] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { withAction, withoutAction } = buildProjectionSeries(nba.projected_impact);

  async function handleApprove() {
    try {
      setError(null);
      const res = await postHitl("approve", {
        client_id: clientId,
        nba_title: nba.title,
        nba_index: index,
      });
      console.log("[HITL] approve →", res);
      setStatus("approved");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleSaveEdit() {
    try {
      setError(null);
      const res = await postHitl("edit", {
        client_id: clientId,
        nba_title: nba.title,
        nba_index: index,
        edited_text: editedText,
      });
      console.log("[HITL] edit →", res);
      setStatus("edited");
      setIsEditing(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleRegenerate() {
    try {
      setError(null);
      setStatus("regenerating");
      const res = await postHitl("regenerate", {
        client_id: clientId,
        nba_title: nba.title,
        nba_index: index,
      });
      console.log("[HITL] regenerate →", res);
      // Brief spinner: reset to idle after 1s (no real per-NBA regen)
      setTimeout(() => setStatus("idle"), 1000);
    } catch (e) {
      setError((e as Error).message);
      setStatus("idle");
    }
  }

  async function handleReject() {
    if (!showRejectInput) {
      setShowRejectInput(true);
      return;
    }
    try {
      setError(null);
      const res = await postHitl("reject", {
        client_id: clientId,
        nba_title: nba.title,
        nba_index: index,
        reason: rejectReason || "No reason given",
      });
      console.log("[HITL] reject →", res);
      setStatus("rejected");
      setShowRejectInput(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const isRejected = status === "rejected";
  const isApproved = status === "approved";
  const isEdited = status === "edited";
  const isRegenerating = status === "regenerating";

  return (
    <div
      className={`bg-paper border border-gray-300 rounded-[14px] p-6 transition-opacity ${
        isRejected ? "opacity-50" : "opacity-100"
      }`}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-start gap-2 mb-3">
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wide border ${PRIORITY_CLASSES[nba.suggested_priority]}`}
        >
          {nba.suggested_priority}
        </span>
        <span className="font-mono text-[10px] text-gray-500 self-center">
          NBA {index + 1}
        </span>
        <div className="ml-auto flex gap-1.5 flex-wrap justify-end">
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono border ${CONFIDENCE_CLASSES[nba.confidence]}`}
          >
            {CONFIDENCE_LABELS[nba.confidence]}
          </span>
          {isApproved && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono bg-olive/20 text-olive border border-olive/40">
              Endorsed
            </span>
          )}
          {isEdited && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono bg-amber/20 text-amber-dark border border-amber/40">
              Edited
            </span>
          )}
          {isRejected && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono bg-rust/20 text-rust border border-rust/40 line-through">
              Rejected
            </span>
          )}
        </div>
      </div>

      {/* Title */}
      <h3
        className={`font-serif text-slate text-lg font-semibold leading-snug mb-2 ${
          isRejected ? "line-through text-gray-500" : ""
        }`}
      >
        {nba.title}
      </h3>

      {/* Rationale */}
      {isEditing ? (
        <textarea
          value={editedText}
          onChange={(e) => setEditedText(e.target.value)}
          rows={4}
          className="w-full rounded-lg border border-clay/40 bg-ivory px-3 py-2 text-sm font-serif text-slate focus:outline-none focus:ring-2 focus:ring-clay/30 resize-none mb-2"
        />
      ) : (
        <p className="text-sm text-gray-700 font-serif leading-relaxed mb-2">
          {isEdited ? editedText : nba.rationale}
        </p>
      )}

      {/* Projected impact */}
      <p className="font-mono text-[11px] text-gray-500 mb-3">
        Projected impact:{" "}
        <span className="text-olive font-semibold">{nba.projected_impact}</span>
      </p>

      {/* Evidence link */}
      {nba.evidence_refs.length > 0 && (
        <button
          onClick={() => onCite(nba.evidence_refs)}
          className="text-xs font-mono text-clay hover:underline mb-4"
        >
          {nba.evidence_refs.length} evidence source
          {nba.evidence_refs.length !== 1 ? "s" : ""} →
        </button>
      )}

      {/* Projection chart */}
      {!isRejected && (
        <div className="mb-4">
          <p className="font-mono text-[10px] uppercase tracking-widest text-gray-500 mb-1">
            12-month projection
          </p>
          <NbaProjection
            withAction={withAction}
            withoutAction={withoutAction}
            height={220}
          />
        </div>
      )}

      {/* Error display */}
      {error && (
        <p className="text-xs font-mono text-rust mb-2">Error: {error}</p>
      )}

      {/* Reject reason input */}
      {showRejectInput && (
        <div className="mb-3 flex gap-2">
          <input
            type="text"
            placeholder="Reason for rejection…"
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            className="flex-1 rounded-lg border border-rust/40 bg-ivory px-3 py-1.5 text-sm font-serif text-slate focus:outline-none focus:ring-2 focus:ring-rust/20"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleReject();
              if (e.key === "Escape") {
                setShowRejectInput(false);
                setRejectReason("");
              }
            }}
          />
          <button
            onClick={handleReject}
            className="px-3 py-1.5 rounded-lg bg-rust/10 text-rust border border-rust/30 text-xs font-mono hover:bg-rust/20 transition-colors"
          >
            Confirm
          </button>
          <button
            onClick={() => {
              setShowRejectInput(false);
              setRejectReason("");
            }}
            className="px-3 py-1.5 rounded-lg bg-gray-100 text-gray-500 border border-gray-300 text-xs font-mono hover:bg-gray-150 transition-colors"
          >
            Cancel
          </button>
        </div>
      )}

      {/* HITL controls */}
      {!isRejected && (
        <div className="flex flex-wrap gap-2 pt-3 border-t border-gray-300">
          {/* Approve */}
          {!isApproved && (
            <button
              onClick={handleApprove}
              disabled={isRegenerating}
              className="px-3 py-1.5 rounded-lg bg-olive/10 text-olive border border-olive/30 text-xs font-mono hover:bg-olive/20 transition-colors disabled:opacity-50"
            >
              Approve
            </button>
          )}

          {/* Edit / Save */}
          {!isEditing ? (
            <button
              onClick={() => setIsEditing(true)}
              disabled={isRegenerating}
              className="px-3 py-1.5 rounded-lg bg-amber/10 text-amber-dark border border-amber/30 text-xs font-mono hover:bg-amber/20 transition-colors disabled:opacity-50"
            >
              Edit
            </button>
          ) : (
            <>
              <button
                onClick={handleSaveEdit}
                className="px-3 py-1.5 rounded-lg bg-olive/10 text-olive border border-olive/30 text-xs font-mono hover:bg-olive/20 transition-colors"
              >
                Save
              </button>
              <button
                onClick={() => {
                  setIsEditing(false);
                  setEditedText(nba.rationale);
                }}
                className="px-3 py-1.5 rounded-lg bg-gray-100 text-gray-500 border border-gray-300 text-xs font-mono hover:bg-gray-150 transition-colors"
              >
                Cancel
              </button>
            </>
          )}

          {/* Regenerate */}
          <button
            onClick={handleRegenerate}
            disabled={isRegenerating || isEditing}
            className="px-3 py-1.5 rounded-lg bg-gray-100 text-gray-700 border border-gray-300 text-xs font-mono hover:bg-gray-150 transition-colors disabled:opacity-50 flex items-center gap-1.5"
          >
            {isRegenerating ? (
              <>
                <span className="inline-block w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                Regenerating…
              </>
            ) : (
              "Regenerate"
            )}
          </button>

          {/* Reject */}
          {!showRejectInput && (
            <button
              onClick={handleReject}
              disabled={isRegenerating}
              className="px-3 py-1.5 rounded-lg bg-rust/10 text-rust border border-rust/30 text-xs font-mono hover:bg-rust/20 transition-colors disabled:opacity-50"
            >
              Reject
            </button>
          )}
        </div>
      )}
    </div>
  );
}
