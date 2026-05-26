/**
 * ChatPanel — Advisor Q&A chat column (Task 4.4).
 *
 * Props:
 *   clientId  — e.g. "bergstrom"
 *   onCite    — called with cited refs when a citation pill is clicked
 *
 * Features:
 *   - 3 suggested prompt chips (pre-filled on click)
 *   - Message list (user + assistant turns), scrolls to bottom
 *   - Assistant messages: answer text + citation pills + confidence badge
 *   - Input box + Send button
 *   - "Thinking…" indicator while awaiting
 *   - Inline error on failure
 */
import { useState, useRef, useEffect } from "react";
import type { KeyboardEvent } from "react";
import { postChat } from "../api";
import type { EvidenceRef, ChatResponse, Confidence } from "../types";

// ── Types ─────────────────────────────────────────────────────────────────────

interface UserMessage {
  role: "user";
  text: string;
}

interface AssistantMessage {
  role: "assistant";
  answer: string;
  cited_refs: EvidenceRef[];
  confidence: Confidence;
}

type Message = UserMessage | AssistantMessage;

interface Props {
  clientId: string;
  onCite: (refs: EvidenceRef[]) => void;
}

// ── Suggested prompts ─────────────────────────────────────────────────────────

const SUGGESTED_PROMPTS = [
  "What if Brent drops to $60?",
  "Any Gulf real estate news worth raising?",
  "What's our ECB-rate exposure?",
];

// ── Confidence badge ──────────────────────────────────────────────────────────

const CONFIDENCE_STYLES: Record<Confidence, string> = {
  high: "bg-olive/10 text-olive border-olive/20",
  medium: "bg-amber/10 text-amber border-amber/20",
  low_needs_verification: "bg-rust/10 text-rust border-rust/20",
};

const CONFIDENCE_LABELS: Record<Confidence, string> = {
  high: "High",
  medium: "Medium",
  low_needs_verification: "Verify",
};

function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-wide border ${CONFIDENCE_STYLES[confidence]}`}
    >
      {CONFIDENCE_LABELS[confidence]}
    </span>
  );
}

// ── Citation pill ─────────────────────────────────────────────────────────────

function CitationPill({
  ref: r,
  onClick,
}: {
  ref: EvidenceRef;
  onClick: () => void;
}) {
  // Show domain from source_uri if available, otherwise doc_id (truncated)
  let label = r.doc_id;
  if (r.source_uri) {
    try {
      label = new URL(r.source_uri).hostname.replace(/^www\./, "");
    } catch {
      label = r.doc_id;
    }
  }
  if (label.length > 28) label = label.slice(0, 28) + "…";

  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-clay/8 border border-clay/20 hover:bg-clay/15 transition-colors text-[10px] font-mono text-clay cursor-pointer"
      title={r.source_uri ?? r.doc_id}
    >
      <span className="text-clay/60">↗</span>
      {label}
    </button>
  );
}

// ── Message bubbles ───────────────────────────────────────────────────────────

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] bg-clay/10 border border-clay/15 rounded-[10px] rounded-tr-sm px-3 py-2">
        <p className="font-serif text-xs text-slate leading-snug">{text}</p>
      </div>
    </div>
  );
}

function AssistantBubble({
  msg,
  onCite,
}: {
  msg: AssistantMessage;
  onCite: (refs: EvidenceRef[]) => void;
}) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[92%] space-y-2">
        {/* Answer text */}
        <div className="bg-paper border border-gray-200 rounded-[10px] rounded-tl-sm px-3 py-2.5">
          <p className="font-serif text-xs text-slate leading-relaxed whitespace-pre-wrap">
            {msg.answer}
          </p>
        </div>

        {/* Footer: confidence badge + citation pills */}
        <div className="flex flex-wrap items-center gap-1.5 px-0.5">
          <ConfidenceBadge confidence={msg.confidence} />
          {msg.cited_refs.slice(0, 6).map((ref, i) => (
            <CitationPill
              key={i}
              ref={ref}
              onClick={() => onCite(msg.cited_refs)}
            />
          ))}
          {msg.cited_refs.length > 6 && (
            <button
              onClick={() => onCite(msg.cited_refs)}
              className="text-[9px] font-mono text-gray-400 hover:text-clay transition-colors"
            >
              +{msg.cited_refs.length - 6} more
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="flex justify-start">
      <div className="bg-paper border border-gray-200 rounded-[10px] rounded-tl-sm px-3 py-2.5">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-clay/40 animate-bounce [animation-delay:0ms]" />
          <span className="w-1.5 h-1.5 rounded-full bg-clay/40 animate-bounce [animation-delay:150ms]" />
          <span className="w-1.5 h-1.5 rounded-full bg-clay/40 animate-bounce [animation-delay:300ms]" />
          <span className="font-mono text-[10px] text-gray-400 ml-1">thinking…</span>
        </div>
      </div>
    </div>
  );
}

// ── Main ChatPanel ────────────────────────────────────────────────────────────

export default function ChatPanel({ clientId, onCite }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestionsVisible, setSuggestionsVisible] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on new messages or thinking state
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || thinking) return;

    setError(null);
    setSuggestionsVisible(false);
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    setThinking(true);

    try {
      const response: ChatResponse = await postChat({
        client_id: clientId,
        question: trimmed,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          answer: response.answer,
          cited_refs: response.cited_refs,
          confidence: response.confidence,
        },
      ]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Request failed";
      setError(msg);
    } finally {
      setThinking(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="bg-paper border border-gray-300 rounded-[14px] flex flex-col sticky top-4 max-h-[calc(100vh-6rem)]">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-gray-200 flex-shrink-0">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-clay mb-0.5">
          Chat
        </p>
        <h3 className="font-serif text-slate text-sm font-semibold leading-tight">
          Advisor Q&amp;A
        </h3>
      </div>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-[200px]">
        {/* Suggested prompts — visible until first message */}
        {suggestionsVisible && messages.length === 0 && (
          <div className="space-y-2 pb-1">
            <p className="font-mono text-[9px] uppercase tracking-widest text-gray-400">
              Try asking
            </p>
            {SUGGESTED_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                onClick={() => sendMessage(prompt)}
                className="block w-full text-left px-3 py-2 rounded-lg border border-clay/20 bg-clay/4 hover:bg-clay/10 hover:border-clay/40 transition-colors"
              >
                <p className="font-serif text-xs text-slate leading-snug">{prompt}</p>
              </button>
            ))}
          </div>
        )}

        {/* Message history */}
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <UserBubble key={i} text={msg.text} />
          ) : (
            <AssistantBubble key={i} msg={msg} onCite={onCite} />
          )
        )}

        {/* Thinking indicator */}
        {thinking && <ThinkingBubble />}

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-rust/30 bg-rust/5 px-3 py-2">
            <p className="font-mono text-[10px] uppercase text-rust mb-0.5">Error</p>
            <p className="font-serif text-xs text-slate">{error}</p>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="px-3 pb-3 pt-2 border-t border-gray-200 flex-shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the brief…"
            disabled={thinking}
            className="flex-1 rounded-lg border border-gray-300 bg-ivory px-3 py-2 text-xs font-serif text-slate placeholder:text-gray-400 focus:outline-none focus:border-clay/60 transition-colors disabled:opacity-50"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={thinking || !input.trim()}
            className="px-3 py-2 rounded-lg bg-clay text-white text-[10px] font-mono uppercase tracking-wide hover:bg-clay/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
