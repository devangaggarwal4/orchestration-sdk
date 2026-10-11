import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { ChartCard, Meter, StatTile, makeTooltip } from "../components/charts";
import { Card, ErrorNote, Select, Spinner } from "../components/ui";
import { compact, ms, pct, usd, usdTick } from "../lib/format";
import { useChartTokens } from "../lib/theme";

// Shapes of /api/stats (built by hybrid_router.reporting.report.build_report).
export interface Stats {
  overview: {
    routes: number;
    calls: number;
    local_route_share: number | null;
    local_token_share: number | null;
    tokens_local: number;
    tokens_cloud: number;
    actual_cost_usd: number;
    counterfactual_cost_usd: number;
    savings_usd: number;
    savings_pct: number | null;
    escalation_rate: number | null;
    prices_verified: boolean | null;
    routes_with_unknown_cost: number;
  };
  by_zone: {
    zone: string;
    routes: number;
    local_share: number | null;
    escalation_rate: number | null;
    cost_usd: number;
    counterfactual_usd: number;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
  }[];
  by_task_type: { task_type: string; routes: number; escalation_rate: number | null; local_share: number | null }[];
  top_escalation_reasons: { reason: string; count: number }[];
  cascade_waste: { wasted_calls: number; local_calls: number; latency_ms: number; tokens: number; wasted_share_of_local: number | null };
  latency_by_tier: { tier: string; calls: number; p50_ms: number | null; p95_ms: number | null; ttft_p50_ms: number | null }[];
  pricing_drift: { reported_usd: number; pricing_yaml_usd: number; ratio: number | null } | null;
  timeseries: { bucket_s: number; points: { ts: number; local: number; cloud: number; actual: number; cf: number }[] };
  score_histogram: { lo: number; hi: number; local_ok: number; escalated: number; cloud_direct: number }[];
  budget: { daily_cloud_budget_usd: number | null; spent_today_usd: number | null };
}

const RANGES = [
  ["24h", "Last 24 hours"],
  ["7d", "Last 7 days"],
  ["30d", "Last 30 days"],
  ["90d", "Last 90 days"],
  ["", "All time"],
] as const;

const BAR = 24; // max bar thickness

export function cumulativeCost(points: { ts: number; actual: number; cf: number }[], bucketS = 0) {
  // Start from zero one bucket earlier so even a single bucket draws a line.
  const first = points[0];
  const out: { ts: number; actual: number; cf: number }[] =
    first && bucketS ? [{ ts: first.ts - bucketS, actual: 0, cf: 0 }] : [];
  for (const p of points) {
    const prev = out.at(-1);
    out.push({ ts: p.ts, actual: (prev?.actual ?? 0) + p.actual, cf: (prev?.cf ?? 0) + p.cf });
  }
  return out;
}
const AXIS = { fontSize: 11 };

export function DashboardPage() {
  const [since, setSince] = useState("7d");
  const [source, setSource] = useState<"main" | "eval">("main");
  const t = useChartTokens();
  const stats = useQuery({
    queryKey: ["stats", since, source],
    queryFn: () => api<Stats>(`/api/stats?since=${since}&source=${source}`),
    refetchInterval: 30_000,
  });
  const cfg = useQuery({
    queryKey: ["config"],
    queryFn: () => api<{ config: { routing: { easy_threshold: number; hard_threshold: number; opus_threshold: number } } }>("/api/config"),
  });

  const s = stats.data;
  const dim = stats.isFetching && !!s;
  const axis = { stroke: t.axis, tick: { ...AXIS, fill: t.muted }, tickLine: false };
  const grid = <CartesianGrid stroke={t.grid} vertical={false} />;
  const timeFmt = (ts: unknown) =>
    new Date(Number(ts) * 1000).toLocaleString(undefined, s && s.timeseries.bucket_s < 86400
      ? { month: "short", day: "numeric", hour: "numeric" }
      : { month: "short", day: "numeric" });

  const cumulative = cumulativeCost(s?.timeseries.points ?? [], s?.timeseries.bucket_s);
  const hist = (s?.score_histogram ?? []).map((b) => ({ ...b, bin: b.lo.toFixed(2) }));
  const r = cfg.data?.config.routing;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3" role="group" aria-label="Filters">
        <Select value={since} onChange={(e) => setSince(e.target.value)} aria-label="Time range">
          {RANGES.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </Select>
        <Select value={source} onChange={(e) => setSource(e.target.value as "main" | "eval")} aria-label="Data source">
          <option value="main">Live traffic</option>
          <option value="eval">Eval runs</option>
        </Select>
        {stats.isFetching && <Spinner label="Refreshing" />}
      </div>
      <ErrorNote error={stats.error} />
      {s && s.overview.routes === 0 && (
        <Card>
          <p className="text-sm text-ink-2">No routed requests in this window yet. Chat, run the agent, or run an eval to see data here.</p>
        </Card>
      )}
      {s && s.overview.routes > 0 && (
        <>
          <div className={`grid gap-3 sm:grid-cols-2 lg:grid-cols-4 ${dim ? "opacity-60" : ""}`}>
            <StatTile
              hero
              label="Saved vs all-cloud"
              value={usd(s.overview.savings_usd)}
              detail={
                <>
                  {pct(s.overview.savings_pct)} of {usd(s.overview.counterfactual_cost_usd)} · spent{" "}
                  {usd(s.overview.actual_cost_usd)}
                  {s.overview.prices_verified === false && " · estimate (prices unverified)"}
                </>
              }
            />
            <StatTile
              label="Served locally"
              value={pct(s.overview.local_route_share)}
              detail={`${compact(s.overview.routes)} routes · ${pct(s.overview.local_token_share)} of tokens`}
            />
            <StatTile
              label="Escalation rate"
              value={pct(s.overview.escalation_rate)}
              detail={`cascade waste ${ms(s.cascade_waste.latency_ms)}, ${compact(s.cascade_waste.tokens)} tokens`}
            />
            <StatTile
              label="Cloud spend today"
              value={usd(s.budget.spent_today_usd ?? 0)}
              detail={
                s.budget.daily_cloud_budget_usd != null
                  ? `of ${usd(s.budget.daily_cloud_budget_usd, 2)} daily budget`
                  : "no daily budget set"
              }
            >
              {s.budget.daily_cloud_budget_usd != null && (
                <Meter value={s.budget.spent_today_usd ?? 0} max={s.budget.daily_cloud_budget_usd} />
              )}
            </StatTile>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard
              title="Routes over time"
              subtitle="where each request was finally served"
              dimmed={dim}
              legend={[
                { label: "Local", color: t["tier-local"] },
                { label: "Cloud", color: t["tier-sonnet"] },
              ]}
              rows={s.timeseries.points}
              columns={[
                { key: "ts", label: "Period", value: (p) => timeFmt(p.ts) },
                { key: "local", label: "Local", value: (p) => p.local, numeric: true },
                { key: "cloud", label: "Cloud", value: (p) => p.cloud, numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={s.timeseries.points} barCategoryGap="20%">
                  {grid}
                  <XAxis dataKey="ts" tickFormatter={timeFmt} {...axis} />
                  <YAxis allowDecimals={false} width={36} {...axis} axisLine={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => `${v}`, timeFmt)} />
                  <Bar dataKey="local" name="Local" stackId="a" fill={t["tier-local"]} stroke={t.surface} strokeWidth={2} maxBarSize={BAR} />
                  <Bar dataKey="cloud" name="Cloud" stackId="a" fill={t["tier-sonnet"]} stroke={t.surface} strokeWidth={2} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Cumulative cost"
              subtitle="actual vs if everything went to the cloud"
              dimmed={dim}
              legend={[
                { label: "Actual", color: t["tier-local"], shape: "line" },
                { label: "All-cloud", color: t["tier-sonnet"], shape: "line" },
              ]}
              rows={cumulative}
              columns={[
                { key: "ts", label: "Period", value: (p) => timeFmt(p.ts) },
                { key: "a", label: "Actual", value: (p) => usd(p.actual), numeric: true },
                { key: "c", label: "All-cloud", value: (p) => usd(p.cf), numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={cumulative}>
                  {grid}
                  <XAxis dataKey="ts" tickFormatter={timeFmt} {...axis} />
                  <YAxis tickFormatter={usdTick} width={56} {...axis} axisLine={false} />
                  <Tooltip cursor={{ stroke: t.axis }} content={makeTooltip((v) => usd(v), timeFmt)} />
                  <Line dataKey="actual" name="Actual" stroke={t["tier-local"]} strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: t.surface, strokeWidth: 2 }} />
                  <Line dataKey="cf" name="All-cloud" stroke={t["tier-sonnet"]} strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: t.surface, strokeWidth: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Score distribution"
              subtitle="by outcome, with current zone thresholds"
              dimmed={dim}
              legend={[
                { label: "Local, passed", color: t["tier-local"] },
                { label: "Cloud directly", color: t["tier-sonnet"] },
                { label: "Escalated", color: t["outcome-escalated"] },
              ]}
              rows={hist}
              columns={[
                { key: "bin", label: "Score", value: (b) => `${b.lo.toFixed(2)}–${b.hi.toFixed(2)}` },
                { key: "l", label: "Local, passed", value: (b) => b.local_ok, numeric: true },
                { key: "c", label: "Cloud directly", value: (b) => b.cloud_direct, numeric: true },
                { key: "e", label: "Escalated", value: (b) => b.escalated, numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={hist} barCategoryGap={2}>
                  {grid}
                  <XAxis dataKey="bin" {...axis} interval={3} />
                  <YAxis allowDecimals={false} width={36} {...axis} axisLine={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => `${v}`, (l) => `score ${l}`)} />
                  <Bar dataKey="local_ok" name="Local, passed" stackId="h" fill={t["tier-local"]} stroke={t.surface} strokeWidth={2} />
                  <Bar dataKey="cloud_direct" name="Cloud directly" stackId="h" fill={t["tier-sonnet"]} stroke={t.surface} strokeWidth={2} />
                  <Bar dataKey="escalated" name="Escalated" stackId="h" fill={t["outcome-escalated"]} stroke={t.surface} strokeWidth={2} radius={[4, 4, 0, 0]} />
                  {r &&
                    (
                      [
                        [r.easy_threshold, "easy"],
                        [r.hard_threshold, "hard"],
                        [r.opus_threshold, "opus"],
                      ] as const
                    ).map(([x, label]) => (
                      <ReferenceLine
                        key={label}
                        x={(Math.floor(x * 20) / 20).toFixed(2)}
                        stroke={t.muted}
                        label={{ value: label, position: "insideTopRight", fill: t.muted, fontSize: 11 }}
                      />
                    ))}
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Escalation rate by task type"
              dimmed={dim}
              rows={s.by_task_type}
              columns={[
                { key: "t", label: "Task type", value: (x) => x.task_type },
                { key: "n", label: "Routes", value: (x) => x.routes, numeric: true },
                { key: "e", label: "Escalated", value: (x) => pct(x.escalation_rate), numeric: true },
                { key: "l", label: "Served locally", value: (x) => pct(x.local_share), numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={Math.max(160, s.by_task_type.length * 30 + 30)}>
                <BarChart data={s.by_task_type} layout="vertical" margin={{ right: 40 }}>
                  <CartesianGrid stroke={t.grid} horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axis} />
                  <YAxis type="category" dataKey="task_type" width={140} {...axis} axisLine={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => pct(v))} />
                  <Bar
                    dataKey="escalation_rate"
                    name="Escalation rate"
                    fill={t["tier-local"]}
                    maxBarSize={18}
                    radius={[0, 4, 4, 0]}
                    label={{ position: "right", fill: t["ink-2"], fontSize: 11, formatter: (v: unknown) => pct(Number(v)) }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Cost by zone"
              dimmed={dim}
              legend={[
                { label: "Actual", color: t["tier-local"] },
                { label: "All-cloud", color: t["tier-sonnet"] },
              ]}
              rows={s.by_zone}
              columns={[
                { key: "z", label: "Zone", value: (z) => z.zone },
                { key: "n", label: "Routes", value: (z) => z.routes, numeric: true },
                { key: "a", label: "Actual", value: (z) => usd(z.cost_usd), numeric: true },
                { key: "c", label: "All-cloud", value: (z) => usd(z.counterfactual_usd), numeric: true },
                { key: "l", label: "Local", value: (z) => pct(z.local_share), numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={s.by_zone} barGap={2}>
                  {grid}
                  <XAxis dataKey="zone" {...axis} />
                  <YAxis tickFormatter={usdTick} width={56} {...axis} axisLine={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => usd(v))} />
                  <Bar dataKey="cost_usd" name="Actual" fill={t["tier-local"]} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="counterfactual_usd" name="All-cloud" fill={t["tier-sonnet"]} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Latency by zone"
              subtitle="end-to-end per request"
              dimmed={dim}
              legend={[
                { label: "p50", color: t["ord-light"] },
                { label: "p95", color: t["ord-dark"] },
              ]}
              rows={s.by_zone}
              columns={[
                { key: "z", label: "Zone", value: (z) => z.zone },
                { key: "p50", label: "p50", value: (z) => ms(z.latency_p50_ms), numeric: true },
                { key: "p95", label: "p95", value: (z) => ms(z.latency_p95_ms), numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={s.by_zone} barGap={2}>
                  {grid}
                  <XAxis dataKey="zone" {...axis} />
                  <YAxis tickFormatter={(v) => ms(v)} width={52} {...axis} axisLine={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => ms(v))} />
                  <Bar dataKey="latency_p50_ms" name="p50" fill={t["ord-light"]} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="latency_p95_ms" name="p95" fill={t["ord-dark"]} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Latency by tier"
              subtitle="per model call"
              dimmed={dim}
              legend={[
                { label: "p50", color: t["ord-light"] },
                { label: "p95", color: t["ord-dark"] },
              ]}
              rows={s.latency_by_tier}
              columns={[
                { key: "t", label: "Tier", value: (x) => x.tier },
                { key: "n", label: "Calls", value: (x) => x.calls, numeric: true },
                { key: "p50", label: "p50", value: (x) => ms(x.p50_ms), numeric: true },
                { key: "p95", label: "p95", value: (x) => ms(x.p95_ms), numeric: true },
                { key: "ttft", label: "TTFT p50", value: (x) => ms(x.ttft_p50_ms), numeric: true },
              ]}
            >
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={s.latency_by_tier} barGap={2}>
                  {grid}
                  <XAxis dataKey="tier" {...axis} />
                  <YAxis tickFormatter={(v) => ms(v)} width={52} {...axis} axisLine={false} />
                  <Tooltip cursor={{ fill: t.grid, opacity: 0.4 }} content={makeTooltip((v) => ms(v))} />
                  <Bar dataKey="p50_ms" name="p50" fill={t["ord-light"]} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="p95_ms" name="p95" fill={t["ord-dark"]} maxBarSize={BAR} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <Card title="Top escalation reasons" className={dim ? "opacity-60" : ""}>
              {s.top_escalation_reasons.length === 0 ? (
                <p className="text-sm text-muted">No escalations in this window.</p>
              ) : (
                <table className="w-full text-sm">
                  <tbody>
                    {s.top_escalation_reasons.map((x) => (
                      <tr key={x.reason} className="border-t border-line first:border-0">
                        <td className="tnum py-1.5 pr-3 text-right font-semibold">{x.count}</td>
                        <td className="py-1.5 text-ink-2">{x.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {s.pricing_drift?.ratio != null && (
                <p className="mt-3 text-xs text-muted">
                  pricing.yaml estimates are {s.pricing_drift.ratio.toFixed(2)}× what OpenRouter reported.
                </p>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
