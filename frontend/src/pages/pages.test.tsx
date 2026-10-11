import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { RouteDecision } from "../api/types";
import { RouteCard } from "../components/RouteCard";
import { cumulativeCost } from "./Dashboard";
import { diffPatch } from "./Tuning";

const decision = {
  route_id: "abcdef123456",
  step_index: 0,
  strategy: "hybrid",
  score: 0.52,
  zone: "uncertain",
  reasons: ["task_type=code (0.48)"],
  features: {},
  forced: false,
  planned_tier: "local",
  final_tier: "sonnet",
  final_model: "anthropic/claude-sonnet-5.5",
  escalation_reason: "local failed heuristic: refusal",
  success: true,
  total_latency_ms: 3200,
  config_hash: "h1",
  escalated: true,
  tiers_attempted: ["local", "sonnet"],
  attempts: [
    {
      call_id: "c1", attempt_index: 0, tier: "local", model: "qwen3:8b", role: "cascade_attempt",
      buffered: true, ok: false, wasted: true, latency_ms: 900, queue_ms: 0, ttft_ms: 100,
      usage: { input_tokens: 100, output_tokens: 5, cache_read_tokens: 0, cache_write_tokens: 0 },
      validator_results: [
        { name: "structural", ok: true, skipped: false, reason: "", ms: 0 },
        { name: "heuristic", ok: false, severity: "soft", skipped: false, reason: "refusal", ms: 0 },
      ],
    },
    {
      call_id: "c2", attempt_index: 1, tier: "sonnet", model: "anthropic/claude-sonnet-5.5", role: "escalation",
      buffered: false, ok: true, wasted: false, latency_ms: 2300, queue_ms: 0, ttft_ms: 400,
      usage: { input_tokens: 120, output_tokens: 40, cache_read_tokens: 0, cache_write_tokens: 0 },
      validator_results: [{ name: "structural", ok: true, skipped: false, reason: "", ms: 0 }],
    },
  ],
} as unknown as RouteDecision;

describe("RouteCard", () => {
  it("summarizes and expands into reasons, attempts and validators", async () => {
    render(<RouteCard decision={decision} costUsd={0.0021} discardedDrafts={["I'm not sure"]} />);
    expect(screen.getByText("Sonnet")).toBeInTheDocument();
    expect(screen.getByText(/escalated from/)).toBeInTheDocument();
    expect(screen.getByText("$0.0021")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /why/i }));
    expect(screen.getByText("task_type=code (0.48)")).toBeInTheDocument();
    expect(screen.getAllByText(/heuristic: refusal/)).toHaveLength(2); // escalation reason + validator chip
    expect(screen.getByText("Discarded draft 1")).toBeInTheDocument();
  });
});

describe("pure helpers", () => {
  it("accumulates cost", () => {
    expect(cumulativeCost([{ ts: 1, actual: 1, cf: 2 }, { ts: 2, actual: 0.5, cf: 1 }])).toEqual([
      { ts: 1, actual: 1, cf: 2 },
      { ts: 2, actual: 1.5, cf: 3 },
    ]);
  });

  it("builds a minimal config patch", () => {
    const cfg = {
      routing: { strategy: "hybrid", easy_threshold: 0.25, hard_threshold: 0.65, opus_threshold: 0.85, escalation_mode: "original", daily_cloud_budget_usd: null },
      scorer: { weights: { tokens: 0.6 }, sticky_steps: 3, sticky_boost: 0.25, tool_error_boost: 0.2 },
      tiers: {},
    };
    const draft = {
      routing: { strategy: "hybrid", easy_threshold: 0.25, hard_threshold: 0.5, opus_threshold: 0.85, escalation_mode: "original" },
      scorer: { weights: { tokens: 0.6 }, sticky_steps: 3, sticky_boost: 0.25, tool_error_boost: 0.2 },
    };
    expect(diffPatch(draft, cfg)).toEqual({ routing: { hard_threshold: 0.5 } });
  });
});
