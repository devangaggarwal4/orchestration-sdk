// Aliases over the generated OpenAPI types (src/api/schema.d.ts, `pnpm gen:api`).
import type { components } from "./schema";

type S = components["schemas"];

// Fields with server-side defaults are always present in responses, but the
// schema marks them optional; Full<> restores that.
type Full<T> = { [K in keyof T]-?: Exclude<T[K], undefined> };
export type Usage = Full<S["Usage"]>;
export type ValidatorResult = Full<S["ValidatorResult"]>;
export type AttemptRecord = Omit<Full<S["AttemptRecord"]>, "usage" | "validator_results"> & {
  usage: Usage;
  validator_results: ValidatorResult[];
};
export type RouteDecision = Omit<Full<S["RouteDecision"]>, "attempts"> & {
  attempts: AttemptRecord[];
};
export type TuneReport = S["TuneReport"];
export type Suggestion = S["Suggestion"];
export type Tier = "local" | "sonnet" | "opus";
export type Strategy = "hybrid" | "pre_route" | "cascade";
export type EvalStrategy = Strategy | "all_local" | "all_cloud";

export type RouterEvent =
  | S["RouteStartEvent"]
  | S["AttemptStartEvent"]
  | S["AttemptEndEvent"]
  | S["RouteEndEvent"]
  | S["MessageStartEvent"]
  | S["TextDeltaEvent"]
  | S["ThinkingDeltaEvent"]
  | S["ToolUseEvent"]
  | S["MessageEndEvent"];

export type AgentEvent =
  | S["StepStartEvent"]
  | S["RouterEventWrapper"]
  | S["ToolCallEvent"]
  | S["ToolResultEvent"]
  | (S["StepEndEvent"] & { cost_usd?: number; total_cost_usd?: number })
  | S["AgentEndEvent"];

export type ChatEvent = AgentEvent | { type: "saved"; message_ids: string[] };

export interface ContentBlock {
  type: string;
  text?: string;
  name?: string;
  input?: Record<string, unknown>;
  content?: string | { type: string; text?: string }[];
  is_error?: boolean;
  tool_use_id?: string;
  id?: string;
}

export interface StoredMessage {
  id: string;
  idx: number;
  role: "user" | "assistant";
  content: ContentBlock[];
  decisions: (RouteDecision & { cost_usd?: number })[];
  created: number;
}

export interface Conversation {
  id: string;
  title: string;
  created: number;
  updated: number;
  settings: Record<string, unknown>;
  messages?: StoredMessage[];
}

export interface RunInfo {
  id: string;
  kind: "agent" | "compare" | "eval";
  status: string;
  created: number;
  finished: number | null;
  params: Record<string, unknown>;
  summary: Record<string, unknown> | null;
  config_hash: string;
}

export interface Health {
  tiers: { tier: Tier; model: string; ok: boolean; detail: string }[];
  config_hash: string;
  fake: boolean;
  budget: Budget;
}

export interface Budget {
  daily_cloud_budget_usd: number | null;
  spent_today_usd: number | null;
  cloud_allowed: boolean;
}

export interface EvalTaskInfo {
  id: string;
  difficulty: "easy" | "medium" | "hard";
  task_type: string;
  min_tier: Tier;
  prompt: string;
  tools: string[];
}
