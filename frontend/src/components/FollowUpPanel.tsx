/**
 * FollowUpPanel — mocked "after-meeting capture" panel.
 * Disabled textarea + caption explaining the feedback loop.
 */
export default function FollowUpPanel() {
  return (
    <div className="bg-paper border border-gray-300 rounded-[14px] p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-clay mb-1">
        After-meeting capture
      </p>
      <p className="text-xs font-mono text-gray-500 mb-4">
        Notes logged here feed back into next Monday's brief.
      </p>

      <textarea
        disabled
        placeholder="Enter meeting notes, client decisions, action items…"
        rows={5}
        className="w-full rounded-lg border border-gray-300 bg-gray-100 px-4 py-3 text-sm font-serif text-gray-500 placeholder:text-gray-300 resize-none cursor-not-allowed"
      />

      <p className="mt-2 text-xs font-mono text-gray-500 italic">
        This would feed back into next Monday's brief (mocked in POC).
      </p>
    </div>
  );
}
