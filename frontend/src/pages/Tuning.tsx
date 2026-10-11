import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Bar, BarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, post } from "../api/client";
import type { TuneReport } from "../api/types";
import { makeTooltip } from "../components/charts";
import { Button, Card, ErrorNote, Select, Spinner } from "../components/ui";
import { ms, signed, usd } from "../lib/format";
import { useChartTokens } from "../lib/theme";
import type { Stats } from "./Dashboard";

interface Routing {
  strategy: string;
  easy_threshold: number;
  hard_threshold: number;
  opus_threshold: number;
  escalation_mode: string;
  daily_cloud_budget_usd: number | null;
}
interface Scorer {
  weights: Record<string, number>;
  sticky_steps: number;
  sticky_boost: number;
  tool_error_boost: number;
}
interface ConfigResp {
  path: string;
  exists: boolean;
  hash: string;
  config: { routing: Routing; scorer: Scorer; tiers: Record<string, { model: string }> };
}
interface WhatIf {
  routes_considered: number;
  routes_changed: number;
  transitions: Record<string, number>;
  est_cost_delta_usd: number;
  est_latency_delta_ms: number;
  est_new_under_escalations: number;
  observed_changes: number;
  estimated_changes: number;
}

interface Draft {
  routing: Pick<Routing, "easy_threshold" | "hard_threshold" | "opus_threshold" | "strategy" | "escalation_mode">;
  scorer: Pick<Scorer, "weights" | "sticky_steps" | "sticky_boost" | "tool_error_boost">;
}

const FEATURES = ["tokens", "tools", "task_type", "keywords", "schema"] as const;

function useDebounced<T>(value: T, delay: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return v;
}

function Slider({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <label className="block text-sm">
      <span className="flex justify-between text-ink-2">
        <span>{label}</span>
        <span className="tnum font-medium text-ink">{value.toFixed(step < 1 ? 2 : 0)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
        aria-label={label}
      />
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </label>
  );
}

/** Patch containing only what differs from the live config. */
export function diffPatch(draft: Draft, cfg: ConfigResp["config"]): Record<string, Record<string, unknown>> {
  const patch: Record<string, Record<string, unknown>> = {};
  for (const [k, v] of Object.entries(draft.routing))
    if (cfg.routing[k as keyof Routing] !== v) (patch.routing ??= {})[k] = v;
  for (const [k, v] of Object.entries(draft.scorer)) {
    const cur = cfg.scorer[k as keyof Scorer];
    if (JSON.stringify(cur) !== JSON.stringify(v)) (patch.scorer ??= {})[k] = v;
  }
  return patch;
}

export function TuningPage() {
  const qc = useQueryClient();
  const t = useChartTokens();
  const [since, setSince] = useState("30d");
  const [edits, setEdits] = useState<{ routing?: Partial<Draft["routing"]>; scorer?: Partial<Draft["scorer"]> }>({});
  const [review, setReview] = useState<{ diff: string } | null>(null);
  const [applied, setApplied] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const cfg = useQuery({ queryKey: ["config"], queryFn: () => api<ConfigResp>("/api/config") });
  const tune = useQuery({ queryKey: ["tune", since], queryFn: () => api<TuneReport>(`/api/tune?since=${since}`) });
  const stats = useQuery({ queryKey: ["stats", since, "main"], queryFn: () => api<Stats>(`/api/stats?since=${since}&source=main`) });

  const c = cfg.data?.config;
  const draft: Draft | null = c
    ? {
        routing: {
          easy_threshold: c.routing.easy_threshold,
          hard_threshold: c.routing.hard_threshold,
          opus_threshold: c.routing.opus_threshold,
          strategy: c.routing.strategy,
          escalation_mode: c.routing.escalation_mode,
          ...edits.routing,
        },
        scorer: {
          weights: c.scorer.weights,
          sticky_steps: c.scorer.sticky_steps,
          sticky_boost: c.scorer.sticky_boost,
          tool_error_boost: c.scorer.tool_error_boost,
          ...edits.scorer,
        },
      }
    : null;
  const patch = draft && c ? diffPatch(draft, c) : {};
  const patchKey = JSON.stringify(patch);
  const debouncedPatch = useDebounced(patchKey, 300);
  const whatif = useQuery({
    queryKey: ["whatif", since, debouncedPatch],
    queryFn: () => post<WhatIf>("/api/config/whatif", { patch: JSON.parse(debouncedPatch), since }),
    enabled: debouncedPatch !== "{}",
  });

  const setRouting = (k: keyof Draft["routing"], v: number | string) => {
    if (!draft) return;
    const r = { ...draft.routing, [k]: v };
    // Keep easy <= hard <= opus while dragging.
    if (k === "easy_threshold") r.hard_threshold = Math.max(r.hard_threshold, r.easy_threshold);
    if (k === "hard_threshold") {
      r.easy_threshold = Math.min(r.easy_threshold, r.hard_threshold);
      r.opus_threshold = Math.max(r.opus_threshold, r.hard_threshold);
    }
    if (k === "opus_threshold") r.hard_threshold = Math.min(r.hard_threshold, r.opus_threshold);
    setEdits((e) => ({ ...e, routing: r }));
    setReview(null);
  };
  const setScorer = (k: keyof Draft["scorer"], v: unknown) => {
    setEdits((e) => ({ ...e, scorer: { ...e.scorer, [k]: v } }));
    setReview(null);
  };
  const applySuggestion = (param: string, value: unknown) => {
    const [section, key] = param.split(".");
    if (section === "routing") setRouting(key as keyof Draft["routing"], value as number);
    if (section === "scorer") setScorer(key as keyof Draft["scorer"], value);
  };

  const doReview = async () => {
    setError(null);
    try {
      setReview(await post<{ diff: string }>("/api/config/validate", { patch }));
    } catch (e) {
      setError(e);
    }
  };
  const doApply = async () => {
    setError(null);
    try {
      const res = await api<{ written: string | null; backup?: string | null }>("/api/config", {
        method: "PUT",
        body: JSON.stringify({ patch }),
      });
      setApplied(res.written ? `Saved to ${res.written}${res.backup ? ` (backup: ${res.backup})` : ""}` : "Applied (demo mode: not written to disk)");
      setEdits({});
      setReview(null);
      await qc.invalidateQueries({ queryKey: ["config"] });
      qc.invalidateQueries({ queryKey: ["health"] });
    } catch (e) {
      setError(e);
    }
  };

  if (!draft || !c) return cfg.error ? <ErrorNote error={cfg.error} /> : <Spinner label="Loading config" />;
  const hist = (stats.data?.score_histogram ?? []).map((b) => ({ ...b, bin: b.lo.toFixed(2) }));
  const changed = patchKey !== "{}";
  const w = whatif.data;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="space-y-4">
        <Card
          title="Zones"
          actions={
            <Select value={since} onChange={(e) => setSince(e.target.value)} aria-label="Replay window">
              <option value="7d">replay last 7 days</option>
              <option value="30d">replay last 30 days</option>
              <option value="90d">replay last 90 days</option>
              <option value="">replay everything</option>
            </Select>
          }
        >
          <div className="space-y-3">
            <Slider label="Easy threshold" value={draft.routing.easy_threshold} onChange={(v) => setRouting("easy_threshold", v)} hint="Below: local, streamed immediately, structural checks only" />
            <Slider label="Hard threshold" value={draft.routing.hard_threshold} onChange={(v) => setRouting("hard_threshold", v)} hint="Above: skip local and go straight to the cloud" />
            <Slider label="Opus threshold" value={draft.routing.opus_threshold} onChange={(v) => setRouting("opus_threshold", v)} hint="Above: Opus instead of Sonnet" />
            <div className="flex flex-wrap gap-3 text-sm text-ink-2">
              <label className="flex items-center gap-1.5">
                Default strategy
                <Select value={draft.routing.strategy} onChange={(e) => setRouting("strategy", e.target.value)}>
                  <option value="hybrid">hybrid</option>
                  <option value="pre_route">pre-route</option>
                  <option value="cascade">cascade</option>
                </Select>
              </label>
              <label className="flex items-center gap-1.5">
                On escalation send
                <Select value={draft.routing.escalation_mode} onChange={(e) => setRouting("escalation_mode", e.target.value)}>
                  <option value="original">the original request</option>
                  <option value="revise">the request + local draft</option>
                </Select>
              </label>
            </div>
          </div>
          {hist.length > 0 && (
            <div className="mt-4">
              <div className="mb-1 text-xs text-muted">Recorded scores; lines show your draft thresholds</div>
              <ResponsiveContainer width="100%" height={140}>
                <BarChart data={hist} barCategoryGap={2}>
                  <XAxis dataKey="bin" interval={3} stroke={t.axis} tick={{ fontSize: 11, fill: t.muted }} tickLine={false} />
                  <YAxis hide allowDecimals={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => `${v}`, (l) => `score ${l}`)} />
                  <Bar dataKey="local_ok" name="Local, passed" stackId="h" fill={t["tier-local"]} stroke={t.surface} strokeWidth={2} />
                  <Bar dataKey="cloud_direct" name="Cloud directly" stackId="h" fill={t["tier-sonnet"]} stroke={t.surface} strokeWidth={2} />
                  <Bar dataKey="escalated" name="Escalated" stackId="h" fill={t["outcome-escalated"]} stroke={t.surface} strokeWidth={2} radius={[4, 4, 0, 0]} />
                  {(["easy_threshold", "hard_threshold", "opus_threshold"] as const).map((k) => (
                    <ReferenceLine key={k} x={(Math.floor(draft.routing[k] * 20) / 20).toFixed(2)} stroke={t.ink} label={{ value: k.split("_")[0], position: "insideTopRight", fill: t["ink-2"], fontSize: 11 }} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Scorer weights">
          <p className="mb-3 text-xs text-muted">
            Each feature independently pushes the score toward hard: score = 1 − Π(1 − weight × feature).
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {FEATURES.map((f) => (
              <Slider
                key={f}
                label={f.replace("_", " ")}
                value={draft.scorer.weights[f] ?? 0}
                onChange={(v) => setScorer("weights", { ...draft.scorer.weights, [f]: v })}
              />
            ))}
            <Slider label="sticky steps" value={draft.scorer.sticky_steps} min={0} max={10} step={1} onChange={(v) => setScorer("sticky_steps", v)} />
            <Slider label="sticky boost" value={draft.scorer.sticky_boost} onChange={(v) => setScorer("sticky_boost", v)} />
            <Slider label="tool error boost" value={draft.scorer.tool_error_boost} onChange={(v) => setScorer("tool_error_boost", v)} />
          </div>
        </Card>
      </div>

      <div className="space-y-4">
        <Card title="What if" actions={whatif.isFetching && <Spinner label="" />}>
          {!changed ? (
            <p className="text-sm text-muted">Move a slider to replay recorded routes under your draft (no model calls).</p>
          ) : w ? (
            <div className="space-y-2 text-sm" data-testid="whatif">
              <p>
                <span className="font-semibold">{w.routes_changed}</span> of {w.routes_considered} recorded routes would
                route differently ({w.observed_changes} observed, {w.estimated_changes} estimated).
              </p>
              <ul className="flex flex-wrap gap-2 text-xs text-ink-2">
                {Object.entries(w.transitions).map(([k, n]) => (
                  <li key={k} className="rounded border border-line px-2 py-0.5">
                    {k}: {n}
                  </li>
                ))}
              </ul>
              <div className="grid grid-cols-3 gap-2 pt-1">
                <div>
                  <div className="text-xs text-ink-2">Cost</div>
                  <div className="tnum text-lg font-semibold">{signed(w.est_cost_delta_usd, (x) => usd(x))}</div>
                </div>
                <div>
                  <div className="text-xs text-ink-2">Latency</div>
                  <div className="tnum text-lg font-semibold">{signed(w.est_latency_delta_ms, ms)}</div>
                </div>
                <div>
                  <div className="text-xs text-ink-2">New under-escalation risk</div>
                  <div className="tnum text-lg font-semibold">{w.est_new_under_escalations.toFixed(1)}</div>
                </div>
              </div>
            </div>
          ) : (
            <ErrorNote error={whatif.error} />
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            <Button disabled={!changed} onClick={() => void doReview()}>
              Review changes
            </Button>
            <Button variant="ghost" disabled={!changed} onClick={() => { setEdits({}); setReview(null); }}>
              Reset draft
            </Button>
          </div>
          {review && (
            <div className="mt-3 space-y-2">
              <pre className="max-h-64 overflow-auto rounded-md bg-surface-2 p-3 text-xs" data-testid="diff">
                {review.diff || "(no file changes)"}
              </pre>
              <Button variant="primary" onClick={() => void doApply()}>
                Apply to {cfg.data?.path.split("/").at(-1)}
              </Button>
            </div>
          )}
          {applied && <p className="mt-2 text-sm text-ink-2"><span aria-hidden className="text-good">✓</span> {applied}</p>}
          <ErrorNote error={error} />
        </Card>

        <Card title="Suggestions from telemetry">
          {tune.isLoading && <Spinner label="Analyzing" />}
          {tune.data && !tune.data.enough_data && <p className="text-sm text-muted">{tune.data.message}</p>}
          {tune.data?.enough_data && tune.data.suggestions.length === 0 && (
            <p className="text-sm text-muted">{tune.data.message}</p>
          )}
          <ul className="space-y-3">
            {tune.data?.suggestions.map((s) => (
              <li key={s.param} className="rounded-lg border border-line p-3 text-sm">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-medium">
                    {s.param}: {JSON.stringify(s.current)} → {JSON.stringify(s.suggested)}
                  </span>
                  <Button className="text-xs" onClick={() => applySuggestion(s.param, s.suggested)}>
                    Apply to draft
                  </Button>
                </div>
                <p className="mt-1 text-ink-2">{s.rationale}</p>
              </li>
            ))}
          </ul>
          {tune.data?.auc_current != null && (
            <p className="mt-3 text-xs text-muted">
              Scorer AUC for predicting local failure: {tune.data.auc_current.toFixed(2)}
              {tune.data.auc_fitted != null && ` (refit ${tune.data.auc_fitted.toFixed(2)})`} over {tune.data.samples} routes.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
