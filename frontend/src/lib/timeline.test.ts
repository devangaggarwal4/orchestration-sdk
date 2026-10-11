import { describe, expect, it } from "vitest";
import { emptyTimeline, reduceTimeline } from "./timeline";

const router = (step: number, event: object) => ({ type: "router", step, event });

describe("reduceTimeline", () => {
  it("builds steps from streamed events and discards rejected drafts", () => {
    const events = [
      { type: "step_start", step: 0, task_type: "chat" },
      router(0, { type: "route_start", score: 0.5, zone: "uncertain", planned_tier: "local" }),
      router(0, { type: "attempt_start", tier: "local", buffered: false }),
      router(0, { type: "text_delta", text: "I'm not " }),
      router(0, { type: "text_delta", text: "sure" }),
      router(0, { type: "attempt_end", discarded: true }),
      router(0, { type: "attempt_start", tier: "sonnet", buffered: false }),
      router(0, { type: "text_delta", text: "Answer" }),
      { type: "tool_call", step: 0, tool_use_id: "t1", name: "read_file", input: { path: "a" } },
      { type: "tool_result", step: 0, tool_use_id: "t1", content: "ok", is_error: false },
      { type: "step_end", step: 0, decision: { final_tier: "sonnet" }, cost_usd: 0.01 },
      { type: "agent_end", status: "done", answer: "Answer", steps: 1 },
      { type: "something_new" },
    ];
    const t = events.reduce((acc, e) => reduceTimeline(acc, e as never), emptyTimeline());
    const s = t.steps[0]!;
    expect(s.score).toBe(0.5);
    expect(s.discarded).toEqual(["I'm not sure"]);
    expect(s.text).toBe("Answer");
    expect(s.currentTier).toBe("sonnet");
    expect(s.tools).toEqual([{ id: "t1", name: "read_file", input: { path: "a" }, result: "ok", isError: false }]);
    expect(t.totalCostUsd).toBeCloseTo(0.01);
    expect(t.end?.status).toBe("done");
  });
});
