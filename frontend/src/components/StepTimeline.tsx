import type { StepView } from "../lib/timeline";
import { usd } from "../lib/format";
import { RouteCard } from "./RouteCard";
import { Spinner, TierBadge } from "./ui";

export function ToolCallView({ name, input, result, isError }: { name: string; input: Record<string, unknown>; result?: string; isError?: boolean }) {
  return (
    <div className="rounded-md border border-line bg-surface text-xs">
      <div className="px-2.5 py-1.5 font-mono text-ink-2">
        → {name}({Object.entries(input).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
      </div>
      {result !== undefined && (
        <pre className="max-h-40 overflow-auto border-t border-line px-2.5 py-1.5 whitespace-pre-wrap text-muted">
          {isError ? "✕ " : "← "}
          {result}
        </pre>
      )}
    </div>
  );
}

export function StepTimeline({ steps, live, compact }: { steps: StepView[]; live?: boolean; compact?: boolean }) {
  return (
    <ol className="space-y-3">
      {steps.map((s, i) => {
        const pending = live && i === steps.length - 1 && !s.decision;
        return (
          <li key={s.step} className="rounded-lg border border-line bg-surface p-3" data-testid="step">
            {!compact && (
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-ink-2">
                <span className="font-medium text-ink">Step {s.step + 1}</span>
                {s.decision ? (
                  <TierBadge tier={s.decision.final_tier} />
                ) : s.currentTier ? (
                  <TierBadge tier={s.currentTier} />
                ) : null}
                {s.score != null && (
                  <span>
                    score <span className="tnum">{s.score.toFixed(2)}</span> · {s.zone}
                  </span>
                )}
                {s.costUsd != null && <span>{usd(s.costUsd)}</span>}
                {pending && <Spinner label={s.buffered ? "validating before showing" : "generating"} />}
              </div>
            )}
            {pending && s.buffered && !s.text && (
              <p className="text-xs text-muted">
                Uncertain zone: the {s.currentTier} answer is checked before it's shown.
              </p>
            )}
            {s.text && <p className="whitespace-pre-wrap text-sm text-ink">{s.text}</p>}
            {s.tools.length > 0 && (
              <div className="mt-2 space-y-1.5">
                {s.tools.map((t) => (
                  <ToolCallView key={t.id} {...t} />
                ))}
              </div>
            )}
            {s.decision && (
              <RouteCard decision={s.decision} costUsd={s.costUsd} discardedDrafts={s.discarded} />
            )}
          </li>
        );
      })}
    </ol>
  );
}
