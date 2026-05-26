/**
 * CitationDrawer — slide-in panel (from the right) showing EvidenceRefs.
 * Opens when refs is non-null; closes via onClose.
 * Includes a CitationSankey built from the refs.
 */
import { useEffect } from "react";
import type { EvidenceRef } from "../types";
import CitationSankey, { type SankeyLink } from "../charts/CitationSankey";

interface Props {
  refs: EvidenceRef[] | null;
  onClose: () => void;
}

function buildSankeyLinks(refs: EvidenceRef[]): SankeyLink[] {
  // Group refs by doc_id; claim label = "Evidence {i+1}"
  const links: SankeyLink[] = [];
  refs.forEach((ref, i) => {
    const source = `Claim ${i + 1}`;
    const target = ref.doc_id.length > 32
      ? ref.doc_id.slice(0, 32) + "…"
      : ref.doc_id;
    links.push({ source, target, value: 1 });
  });
  return links;
}

export default function CitationDrawer({ refs, onClose }: Props) {
  const isOpen = refs !== null && refs.length > 0;

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate/20 z-40 transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Drawer */}
      <div
        className={`fixed top-0 right-0 h-full w-full max-w-md bg-paper border-l border-gray-300 shadow-xl z-50 flex flex-col transition-transform duration-300 ease-out ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
        role="dialog"
        aria-modal="true"
        aria-label="Citation evidence"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-300">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-clay">
              Evidence
            </p>
            <p className="font-serif text-slate text-base font-semibold leading-tight">
              Source Citations
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-slate transition-colors text-xl leading-none p-1"
            aria-label="Close drawer"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {refs && refs.length > 0 && (
            <>
              {/* Sankey */}
              {refs.length >= 2 && (
                <div className="mb-2">
                  <p className="font-mono text-[10px] uppercase tracking-widest text-gray-500 mb-1">
                    Document flow
                  </p>
                  <CitationSankey
                    links={buildSankeyLinks(refs)}
                    height={200}
                  />
                </div>
              )}

              {/* Ref list */}
              <div className="space-y-3">
                {refs.map((ref, i) => (
                  <div
                    key={i}
                    className="bg-gray-100 border border-gray-300 rounded-lg p-3"
                  >
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <span className="font-mono text-[10px] uppercase tracking-wide text-clay">
                        Claim {i + 1}
                      </span>
                      {ref.chunk_id && (
                        <span className="font-mono text-[10px] text-gray-500">
                          chunk: {ref.chunk_id}
                        </span>
                      )}
                    </div>
                    <p className="font-mono text-xs text-gray-700 break-all">
                      {ref.doc_id}
                    </p>
                    {ref.excerpt && (
                      <p className="mt-2 text-sm font-serif text-gray-700 italic leading-snug border-l-2 border-clay/40 pl-2">
                        "{ref.excerpt}"
                      </p>
                    )}
                    {ref.source_uri && (
                      <a
                        href={ref.source_uri}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-1 inline-block text-xs font-mono text-clay hover:underline"
                      >
                        ↗ Source
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
