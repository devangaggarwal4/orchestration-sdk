import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, post, stream } from "../api/client";
import type { EvalStrategy, EvalTaskInfo, RouteDecision } from "../api/types";
import { RouteCard } from "../components/RouteCard";
import { Button, Card, ErrorNote, Select, Spinner, TierBadge } from "../components/ui";
import { ms, pct, usd } from "../lib/format";

const STRATEGIES: { id: EvalStrategy; label: string }[] = [
  { id: "hybrid", label: "Hybrid" },
  { id: "pre_route", label: "Pre-route" },
  { id: "cascade", label: "Cascade" },
  { id: "all_local", label: "All local" },
  { id: "all_cloud", label: "All cloud" },
];

interface Leg {
  text: string;
  done: boolean;
  tiers?: string[];
  cost_usd?: number;
  latency_ms?: number;
  decision?: RouteDecision;
  passed?: boolean;
  failures?: string[];
  error?: string | null;
}

interface EvalRow {
  strategy: string;
  task_id: string;
  passed: boolean;
  skipped: boolean;
  tiers: string[];
  cost_usd: number;
}

interface Summary {
  strategy: string;
  pass_rate: number;
  pass_by_difficulty: Record<string, number>;
  local_share: number;
  routing_accuracy: number | null;
  label_source: string;
  under_escalations: number;
  over_escalations: number;
  cost_usd: number;
  counterfactual_usd: number;
  savings_usd: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  waste_latency_ms: number;
}

type Ev = { type: string; leg?: string; text?: string } & Partial<Leg> & Record<string, unknown>;

function StrategyPicker({ value, onChange }: { value: EvalStrategy[]; onChange: (v: EvalStrategy[]) => void }) {
  return (
    <fieldset className="flex flex-wrap gap-3 text-sm">
      <legend className="sr-only">Strategies</legend>
      {STRATEGIES.map((s) => (
        <label key={s.id} className="flex items-center gap-1.5 text-ink-2">
          <input
            type="checkbox"
            checked={value.includes(s.id)}
            onChange={(e) =>
              onChange(e.target.checked ? [...value, s.id] : value.filter((x) => x !== s.id))
            }
          />
          {s.label}
        </label>
      ))}
    </fieldset>
  );
}

function SideBySide() {
  const tasks = useQuery({ queryKey: ["eval-tasks"], queryFn: () => api<EvalTaskInfo[]>("/api/eval/tasks") });
  const [mode, setMode] = useState<"prompt" | "task">("prompt");
  const [prompt, setPrompt] = useState("");
  const [taskId, setTaskId] = useState("");
  const [strategies, setStrategies] = useState<EvalStrategy[]>(["hybrid", "all_local", "all_cloud"]);
  const [legs, setLegs] = useState<Record<string, Leg>>({});
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const abort = useRef<AbortController | null>(null);
  const task = tasks.data?.find((t) => t.id === taskId);

  const run = async () => {
    setError(null);
    setLegs(Object.fromEntries(strategies.map((s) => [s, { text: "", done: false }])));
    setRunning(true);
    abort.current = new AbortController();
    try {
      const { run_id } = await post<{ run_id: string }>("/api/runs", {
        kind: "compare",
        strategies,
        ...(mode === "prompt" ? { prompt } : { task_id: taskId }),
      });
      await stream<Ev>(
        `/api/runs/${run_id}/events`,
        { method: "GET" },
        (ev) => {
          const leg = ev.leg;
          if (ev.type === "run_error") setError(ev.error);
          if (!leg) return;
          setLegs((all) => {
            const cur = all[leg] ?? { text: "", done: false };
            if (ev.type === "leg_delta") return { ...all, [leg]: { ...cur, text: cur.text + (ev.text ?? "") } };
            if (ev.type === "leg_discard") return { ...all, [leg]: { ...cur, text: "" } };
            if (ev.type === "leg_end")
              return { ...all, [leg]: { ...cur, ...(ev as Partial<Leg>), text: (ev.text as string) ?? cur.text, done: true } };
            return all;
          });
        },
        abort.current.signal,
      );
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(e);
    } finally {
      setRunning(false);
    }
  };

  const ready = strategies.length > 0 && (mode === "prompt" ? prompt.trim() : taskId);
  return (
    <Card title="Side-by-side">
      <div className="space-y-3">
        <div className="flex gap-2 text-sm">
          <Select value={mode} onChange={(e) => setMode(e.target.value as "prompt" | "task")} aria-label="Input">
            <option value="prompt">Your prompt</option>
            <option value="task">Eval task (checked answer)</option>
          </Select>
          {mode === "task" && (
            <Select value={taskId} onChange={(e) => setTaskId(e.target.value)} aria-label="Eval task" className="min-w-64">
              <option value="">Choose a task…</option>
              {tasks.data?.map((t) => (
                <option key={t.id} value={t.id}>
                  [{t.difficulty}] {t.id}
                </option>
              ))}
            </Select>
          )}
        </div>
        {mode === "prompt" ? (
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            aria-label="Prompt"
            placeholder="One prompt, sent under each strategy"
            className="w-full rounded-md border border-line bg-surface px-3 py-2 text-sm"
          />
        ) : (
          task && <p className="rounded-md bg-surface-2 p-3 text-sm whitespace-pre-wrap text-ink-2">{task.prompt}</p>
        )}
        <StrategyPicker value={strategies} onChange={setStrategies} />
        <div className="flex items-center gap-3">
          <Button variant="primary" disabled={!ready || running} onClick={() => void run()}>
            Compare
          </Button>
          {running && <Spinner label="Running (local legs take turns)" />}
        </div>
        <ErrorNote error={error} />
      </div>
      {Object.keys(legs).length > 0 && (
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3" data-testid="legs">
          {Object.entries(legs).map(([name, leg]) => (
            <div key={name} className="rounded-lg border border-line p-3" data-testid={`leg-${name}`}>
              <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                <span className="font-semibold">{STRATEGIES.find((s) => s.id === name)?.label ?? name}</span>
                {leg.tiers?.map((t, i) => <TierBadge key={i} tier={t} />)}
                {!leg.done && <Spinner label="" />}
                {leg.passed != null && (
                  <span className="text-xs">
                    <span aria-hidden className={leg.passed ? "text-good" : "text-bad"}>
                      {leg.passed ? "✓" : "✕"}
                    </span>{" "}
                    {leg.passed ? "passed" : "failed"}
                  </span>
                )}
              </div>
              {leg.done && (
                <div className="mb-2 flex gap-3 text-xs text-ink-2">
                  <span>{usd(leg.cost_usd ?? 0)}</span>
                  <span>{ms(leg.latency_ms)}</span>
                </div>
              )}
              {leg.error && <p className="text-sm text-ink">⚠ {leg.error}</p>}
              {leg.failures && leg.failures.length > 0 && (
                <p className="text-xs text-muted">{leg.failures.join("; ")}</p>
              )}
              <p className="max-h-64 overflow-auto text-sm whitespace-pre-wrap">{leg.text}</p>
              {leg.decision && <RouteCard decision={leg.decision} costUsd={leg.cost_usd} />}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function EvalLauncher() {
  const tasks = useQuery({ queryKey: ["eval-tasks"], queryFn: () => api<EvalTaskInfo[]>("/api/eval/tasks") });
  const [strategies, setStrategies] = useState<EvalStrategy[]>(STRATEGIES.map((s) => s.id));
  const [difficulty, setDifficulty] = useState("");
  const [budget, setBudget] = useState("2");
  const [rows, setRows] = useState<EvalRow[]>([]);
  const [summaries, setSummaries] = useState<Summary[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const subset = (tasks.data ?? []).filter((t) => !difficulty || t.difficulty === difficulty);
  const total = subset.length * strategies.length;

  const run = async () => {
    setRows([]);
    setSummaries([]);
    setError(null);
    setRunning(true);
    try {
      const { run_id } = await post<{ run_id: string }>("/api/runs", {
        kind: "eval",
        strategies,
        only: subset.map((t) => t.id),
        budget_usd: Number(budget),
      });
      await stream<{ type: string } & Record<string, unknown>>(`/api/runs/${run_id}/events`, { method: "GET" }, (ev) => {
        if (ev.type === "eval_result") setRows((r) => [...r, ev as unknown as EvalRow]);
        if (ev.type === "eval_summary") setSummaries(ev.summaries as Summary[]);
        if (ev.type === "run_error") setError(ev.error);
      });
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Card title="Full eval run">
      <div className="space-y-3">
        <StrategyPicker value={strategies} onChange={setStrategies} />
        <div className="flex flex-wrap items-center gap-3 text-sm text-ink-2">
          <Select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} aria-label="Tasks">
            <option value="">All {tasks.data?.length ?? ""} tasks</option>
            <option value="easy">Easy only</option>
            <option value="medium">Medium only</option>
            <option value="hard">Hard only</option>
          </Select>
          <label className="flex items-center gap-1.5">
            Budget $
            <input
              type="number"
              min="0"
              step="0.5"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              className="w-20 rounded-md border border-line bg-surface px-2 py-1"
            />
          </label>
          <Button variant="primary" disabled={running || !strategies.length} onClick={() => void run()}>
            Run {total} task runs
          </Button>
          {running && <Spinner label={`${rows.length} / ${total}`} />}
        </div>
        <ErrorNote error={error} />
      </div>
      {summaries.length > 0 && (
        <div className="mt-4 overflow-x-auto" data-testid="eval-summary">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                {["Strategy", "Pass", "Easy / med / hard", "Local", "Routing acc.", "Under / over", "Cost", "All-cloud", "Saved", "p50 / p95", "Waste"].map((h) => (
                  <th key={h} className="py-1 pr-3 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tnum">
              {summaries.map((s) => (
                <tr key={s.strategy} className="border-t border-line">
                  <td className="py-1.5 pr-3 font-medium">{s.strategy}</td>
                  <td className="pr-3">{pct(s.pass_rate)}</td>
                  <td className="pr-3">
                    {["easy", "medium", "hard"].map((d) => pct(s.pass_by_difficulty[d])).join(" / ")}
                  </td>
                  <td className="pr-3">{pct(s.local_share)}</td>
                  <td className="pr-3">{pct(s.routing_accuracy)}</td>
                  <td className="pr-3">
                    {s.under_escalations} / {s.over_escalations}
                  </td>
                  <td className="pr-3">{usd(s.cost_usd)}</td>
                  <td className="pr-3">{usd(s.counterfactual_usd)}</td>
                  <td className="pr-3">{usd(s.savings_usd)}</td>
                  <td className="pr-3">
                    {ms(s.latency_p50_ms)} / {ms(s.latency_p95_ms)}
                  </td>
                  <td className="pr-3">{ms(s.waste_latency_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">
            Routing accuracy vs {summaries[0]?.label_source}. Under = served locally but failed; over = sent to the
            cloud though local would have passed.
          </p>
        </div>
      )}
      {rows.length > 0 && (
        <details className="mt-3 text-sm" open={summaries.length === 0}>
          <summary className="cursor-pointer text-ink-2">Per-task results ({rows.length})</summary>
          <ul className="mt-2 max-h-72 space-y-0.5 overflow-auto text-xs">
            {rows.map((r, i) => (
              <li key={i} className="flex items-center gap-2">
                <span aria-hidden className={r.passed ? "text-good" : "text-bad"}>
                  {r.skipped ? "–" : r.passed ? "✓" : "✕"}
                </span>
                <span className="w-20 text-ink-2">{r.strategy}</span>
                <span className="w-48 truncate">{r.task_id}</span>
                <span className="text-muted">{r.tiers.join(" → ")}</span>
                <span className="tnum ml-auto text-muted">{usd(r.cost_usd)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Card>
  );
}

export function ComparePage() {
  return (
    <div className="space-y-4">
      <SideBySide />
      <EvalLauncher />
    </div>
  );
}
