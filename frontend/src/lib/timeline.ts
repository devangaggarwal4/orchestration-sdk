import type { AgentEvent, RouteDecision } from "../api/types";

export interface ToolView {
  id: string;
  name: string;
  input: Record<string, unknown>;
  result?: string;
  isError?: boolean;
}

export interface StepView {
  step: number;
  taskType?: string;
  score?: number;
  zone?: string;
  plannedTier?: string;
  currentTier?: string;
  buffered?: boolean;
  text: string;
  thinking: string;
  discarded: string[];
  tools: ToolView[];
  decision?: RouteDecision;
  costUsd?: number;
}

export interface Timeline {
  steps: StepView[];
  end?: { status: string; answer: string; steps: number; error?: string | null };
  totalCostUsd: number;
}

export const emptyTimeline = (): Timeline => ({ steps: [], totalCostUsd: 0 });

function update(t: Timeline, step: number, f: (s: StepView) => StepView): Timeline {
  return { ...t, steps: t.steps.map((s) => (s.step === step ? f(s) : s)) };
}

/** Pure reducer: fold one streamed agent event into the timeline. */
export function reduceTimeline(t: Timeline, ev: AgentEvent | { type: string }): Timeline {
  switch (ev.type) {
    case "step_start": {
      const e = ev as Extract<AgentEvent, { type: "step_start" }>;
      return {
        ...t,
        steps: [
          ...t.steps,
          { step: e.step, taskType: e.task_type, text: "", thinking: "", discarded: [], tools: [] },
        ],
      };
    }
    case "router": {
      const e = ev as Extract<AgentEvent, { type: "router" }>;
      const r = e.event;
      return update(t, e.step, (s) => {
        switch (r.type) {
          case "route_start":
            return { ...s, score: r.score, zone: r.zone, plannedTier: r.planned_tier };
          case "attempt_start":
            return { ...s, currentTier: r.tier, buffered: r.buffered };
          case "text_delta":
            return { ...s, text: s.text + r.text };
          case "thinking_delta":
            return { ...s, thinking: s.thinking + r.thinking };
          case "attempt_end":
            return r.discarded ? { ...s, discarded: [...s.discarded, s.text], text: "" } : s;
          default:
            return s;
        }
      });
    }
    case "tool_call": {
      const e = ev as Extract<AgentEvent, { type: "tool_call" }>;
      return update(t, e.step, (s) => ({
        ...s,
        tools: [...s.tools, { id: e.tool_use_id, name: e.name, input: e.input }],
      }));
    }
    case "tool_result": {
      const e = ev as Extract<AgentEvent, { type: "tool_result" }>;
      return update(t, e.step, (s) => ({
        ...s,
        tools: s.tools.map((x) =>
          x.id === e.tool_use_id ? { ...x, result: e.content, isError: e.is_error } : x,
        ),
      }));
    }
    case "step_end": {
      const e = ev as Extract<AgentEvent, { type: "step_end" }> & { cost_usd?: number };
      const decision = e.decision as RouteDecision;
      const next = update(t, e.step, (s) => ({ ...s, decision, costUsd: e.cost_usd }));
      return { ...next, totalCostUsd: t.totalCostUsd + (e.cost_usd ?? 0) };
    }
    case "agent_end": {
      const e = ev as Extract<AgentEvent, { type: "agent_end" }>;
      return { ...t, end: { status: e.status, answer: e.answer, steps: e.steps, error: e.error } };
    }
    default:
      return t;
  }
}
