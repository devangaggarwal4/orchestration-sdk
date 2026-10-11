import { useState } from "react";
import type { RouteDecision } from "../api/types";
import { ms, usd } from "../lib/format";
import { TierBadge, cx } from "./ui";

/** Why a step went where it did: tier, score, zone, reasons, attempts, validators, cost. */
export function RouteCard({
  decision,
  costUsd,
  discardedDrafts = [],
}: {
  decision: RouteDecision;
  costUsd?: number;
  discardedDrafts?: string[];
}) {
  const [open, setOpen] = useState(false);
  const d = decision;
  const served = d.attempts.find((a) => a.tier === d.final_tier && !a.wasted) ?? d.attempts.at(-1);
  const tokens = served
    ? served.usage.input_tokens + served.usage.cache_read_tokens + served.usage.cache_write_tokens
    : 0;
  return (
    <div className="mt-2 rounded-lg border border-line bg-surface-2/60 text-xs" data-testid="route-card">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 text-left"
      >
        <TierBadge tier={d.final_tier} />
        {d.escalated && (
          <span className="text-ink-2">
            escalated from <span className="font-medium text-ink">{d.attempts[0]?.tier}</span>
          </span>
        )}
        <span className="text-ink-2">
          score <span className="tnum font-medium text-ink">{d.score.toFixed(2)}</span> · {d.zone}
        </span>
        <span className="text-ink-2">{ms(d.total_latency_ms)}</span>
        {costUsd != null && <span className="text-ink-2">{usd(costUsd)}</span>}
        {!d.success && <span className="text-ink">⚠ no attempt passed validation</span>}
        <span className="ml-auto text-muted">{open ? "Hide details" : "Why?"}</span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-line px-3 py-2">
          <div>
            <div className="mb-1 font-medium text-ink">Score reasons</div>
            <ul className="list-inside list-disc text-ink-2">
              {d.reasons.length ? d.reasons.map((r) => <li key={r}>{r}</li>) : <li>no strong signals</li>}
            </ul>
          </div>
          {d.escalation_reason && (
            <div>
              <span className="font-medium text-ink">Escalation: </span>
              <span className="text-ink-2">{d.escalation_reason}</span>
            </div>
          )}
          <div>
            <div className="mb-1 font-medium text-ink">Attempts</div>
            <ol className="space-y-1.5">
              {d.attempts.map((a) => (
                <li key={a.call_id} className="flex flex-wrap items-center gap-2">
                  <TierBadge tier={a.tier} />
                  <span className="text-ink-2">{a.role.replace("_", " ")}</span>
                  <span className="tnum text-ink-2">{ms(a.latency_ms)}</span>
                  {a.queue_ms > 50 && <span className="text-muted">queued {ms(a.queue_ms)}</span>}
                  <span className="tnum text-muted">
                    {a.usage.input_tokens + a.usage.cache_read_tokens}→{a.usage.output_tokens} tok
                  </span>
                  {a.wasted && <span className="text-muted">(discarded)</span>}
                  <span className="flex flex-wrap gap-1">
                    {a.validator_results.map((v) => (
                      <span
                        key={v.name}
                        title={v.reason}
                        className={cx(
                          "rounded border border-line px-1.5 py-0.5",
                          v.skipped ? "text-muted" : "text-ink",
                        )}
                      >
                        <span aria-hidden className={v.ok ? "text-good" : "text-bad"}>
                          {v.skipped ? "–" : v.ok ? "✓" : "✕"}
                        </span>{" "}
                        {v.name}
                        {!v.ok && v.reason ? `: ${v.reason}` : ""}
                      </span>
                    ))}
                  </span>
                  {a.error && <span className="text-ink">error: {a.error}</span>}
                </li>
              ))}
            </ol>
          </div>
          {discardedDrafts.map((t, i) => (
            <details key={i} className="rounded border border-line p-2">
              <summary className="cursor-pointer text-ink-2">Discarded draft {i + 1}</summary>
              <p className="mt-1 whitespace-pre-wrap text-muted">{t || "(no text)"}</p>
            </details>
          ))}
          <div className="text-muted">
            {served?.model} · {tokens} input tokens · route {d.route_id.slice(0, 8)} · config{" "}
            {d.config_hash}
          </div>
        </div>
      )}
    </div>
  );
}
