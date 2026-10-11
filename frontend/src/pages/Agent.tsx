import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, post, stream } from "../api/client";
import type { RunInfo } from "../api/types";
import { StepTimeline } from "../components/StepTimeline";
import { Button, Card, Empty, ErrorNote, Select, Spinner, cx } from "../components/ui";
import { timeAgo, usd } from "../lib/format";
import { emptyTimeline, reduceTimeline, type Timeline } from "../lib/timeline";

type RunEvent = { type: string; status?: string; error?: string; summary?: Record<string, unknown> };

function RunView({ runId }: { runId: string }) {
  const qc = useQueryClient();
  const [timeline, setTimeline] = useState<Timeline>(emptyTimeline);
  const [status, setStatus] = useState<string>("running");
  const [error, setError] = useState<unknown>(null);
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api<RunInfo>(`/api/runs/${runId}`) });

  useEffect(() => {
    const ctrl = new AbortController();
    stream<RunEvent>(
      `/api/runs/${runId}/events`,
      { method: "GET" },
      (ev) => {
        if (ev.type === "run_end") {
          setStatus(ev.status ?? "done");
          qc.invalidateQueries({ queryKey: ["runs", "agent"] });
        } else if (ev.type === "run_error") {
          setError(ev.error);
        } else {
          setTimeline((t) => reduceTimeline(t, ev as Parameters<typeof reduceTimeline>[1]));
        }
      },
      ctrl.signal,
    ).catch((e) => {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(e);
    });
    return () => ctrl.abort();
  }, [runId, qc]);

  const running = status === "running";
  const tiers = timeline.steps.map((s) => s.decision?.final_tier).filter(Boolean);
  const local = tiers.filter((t) => t === "local").length;
  return (
    <div className="space-y-3">
      <Card
        title={String(run.data?.params.task ?? "Agent run")}
        actions={
          running ? (
            <>
              <Spinner label="Running" />
              <Button onClick={() => post(`/api/runs/${runId}/cancel`, {})}>Cancel</Button>
            </>
          ) : (
            <span className="text-xs text-ink-2">{status}</span>
          )
        }
      >
        <div className="flex flex-wrap gap-4 text-sm text-ink-2" data-testid="run-summary">
          <span>
            <span className="font-semibold text-ink">{timeline.steps.length}</span> steps
          </span>
          <span>
            <span className="font-semibold text-ink">{local}</span> local ·{" "}
            <span className="font-semibold text-ink">{tiers.length - local}</span> cloud
          </span>
          <span>
            cost <span className="font-semibold text-ink">{usd(timeline.totalCostUsd)}</span>
          </span>
        </div>
      </Card>
      <ErrorNote error={error} />
      <StepTimeline steps={timeline.steps} live={running} />
      {timeline.end && (
        <Card title={timeline.end.status === "done" ? "Answer" : `Stopped: ${timeline.end.status}`}>
          <p className="text-sm whitespace-pre-wrap" data-testid="agent-answer">
            {timeline.end.answer || timeline.end.error || "(no answer)"}
          </p>
        </Card>
      )}
    </div>
  );
}

export function AgentPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [task, setTask] = useState("");
  const [strategy, setStrategy] = useState("");
  const [forceTier, setForceTier] = useState("");
  const [error, setError] = useState<unknown>(null);
  const runs = useQuery({
    queryKey: ["runs", "agent"],
    queryFn: () => api<RunInfo[]>("/api/runs?kind=agent"),
    refetchInterval: runId ? false : 10_000,
  });

  const start = async () => {
    setError(null);
    try {
      const { run_id } = await post<{ run_id: string }>("/api/runs", {
        kind: "agent",
        task,
        strategy: strategy || null,
        force_tier: forceTier || null,
      });
      qc.invalidateQueries({ queryKey: ["runs", "agent"] });
      navigate(`/agent/${run_id}`);
    } catch (e) {
      setError(e);
    }
  };

  return (
    <div className="grid gap-4 md:grid-cols-[300px_1fr]">
      <aside className="min-w-0 space-y-3">
        <Card title="New agent run">
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            rows={4}
            placeholder="e.g. Where is the escalation logic, and what sends a request to Opus?"
            aria-label="Agent task"
            className="w-full rounded-md border border-line bg-surface px-3 py-2 text-sm"
          />
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            <Select value={strategy} onChange={(e) => setStrategy(e.target.value)} aria-label="Strategy">
              <option value="">default strategy</option>
              <option value="hybrid">hybrid</option>
              <option value="pre_route">pre-route</option>
              <option value="cascade">cascade</option>
            </Select>
            <Select value={forceTier} onChange={(e) => setForceTier(e.target.value)} aria-label="Force tier">
              <option value="">any tier</option>
              <option value="local">local only</option>
              <option value="sonnet">sonnet only</option>
              <option value="opus">opus only</option>
            </Select>
          </div>
          <Button variant="primary" className="mt-3 w-full" disabled={!task.trim()} onClick={() => void start()}>
            Run agent
          </Button>
          <p className="mt-2 text-xs text-muted">
            Tools: read_file and search_text in the configured workspace
            (<code>agent.workspace</code>); run_python only if <code>agent.allow_code_exec</code>.
          </p>
          <ErrorNote error={error} />
        </Card>
        <Card title="Past runs">
          {!runs.data?.length && <p className="text-sm text-muted">None yet.</p>}
          <ul className="space-y-0.5">
            {runs.data?.map((r) => (
              <li key={r.id}>
                <button
                  onClick={() => navigate(`/agent/${r.id}`)}
                  className={cx(
                    "w-full rounded-md px-2 py-1.5 text-left text-sm",
                    r.id === runId ? "bg-surface-2" : "hover:bg-surface-2",
                  )}
                >
                  <span className="line-clamp-1 text-ink">{String(r.params.task ?? "")}</span>
                  <span className="text-xs text-muted">
                    {r.status} · {timeAgo(r.created)}
                    {r.summary?.cost_usd != null && ` · ${usd(Number(r.summary.cost_usd))}`}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
      </aside>
      <section className="min-w-0">
        {runId ? (
          <RunView key={runId} runId={runId} />
        ) : (
          <Empty>Start a run to watch each step get routed live: tier, score, tool calls, escalations and cost.</Empty>
        )}
      </section>
    </div>
  );
}
