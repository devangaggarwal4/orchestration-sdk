# hybrid-router — Implementation Plan

Status: **all eight phases built** (2026-10-11). The user approved building phases 2–8 without stopping between them. The experimental Anthropic-compatible proxy (phase 9) has not been started.

## 0. Decisions already made (from Q&A)

| Topic | Decision |
|---|---|
| Hardware | 16 GB Mac → default local model `qwen3:8b` (Q4_K_M, ~5 GB), `think: false` by default for latency. Embedding model for later: `nomic-embed-text`. Defaults lean conservative (lower `easy_threshold`) because an 8B model will fail more medium tasks. |
| Counterfactual baseline | Every route priced as if sent to one configurable baseline model, default = the Sonnet tier. |
| Counterfactual tokens | Local token counts × a calibrated ratio learned from escalated routes (where we have both local and Claude counts). Falls back to `token_ratio_default` (1.15) until there are ≥ 20 samples. |
| API style | Async core (`await router.acomplete(...)`, `router.astream(...)`) with a sync wrapper (`router.complete(...)`). |
| Web app | Full orchestration web app (see §10): chat playground, live agent runs, live config tuning with what-if replay, side-by-side strategy compare + eval launcher, and the telemetry dashboard. FastAPI backend, React + Vite + TypeScript frontend shipped as built static assets inside the Python package. |
| Tool errors | Approved: an erroring `tool_result` for a locally-served call raises the next step's score and back-fills `local_succeeded=false` on that call. |
| Agent integration | Option C: our own agent loop + LangGraph adapter are first-class. Claude Agent SDK gets the supported "router as a tool" pattern (`delegate_local`) in phase 6; the Anthropic-compatible proxy (`hybrid-router serve`, full streaming, exact pass-through) is an experimental extra after phase 8, documented as unsupported by Anthropic. |
| Cloud provider | Decided 2026-10-11: no Anthropic API key, so the cloud tiers reach Claude through **OpenRouter** with a new `openai_compat` backend (OpenAI-style `/chat/completions`). The `anthropic` backend and dependency were removed. Tiers stay local → sonnet → opus, defaulting to `anthropic/claude-sonnet-5.5` and `anthropic/claude-opus-5.5`. The same backend also serves OpenAI-compatible local runtimes (llama.cpp, LM Studio, vLLM, MLX). |
| Actual cost | The USD OpenRouter reports per call (`usage.cost`) is the actual cost. `pricing.yaml` is used for the counterfactual and as a fallback when no cost is reported. |
| Web deployment | Single user, reachable on LAN / tailnet. Bearer-token auth, required whenever bound to a non-loopback address. |

## 1. Defaults I picked (tell me to change any)

1. **Canonical message format is Anthropic-shaped** (role + list of content blocks: `text`, `tool_use`, `tool_result`, `image`). Ollama's OpenAI-style messages/tool calls are converted inside `OllamaBackend`. Reason: Claude is the quality reference and the Agent SDK speaks this format; converting down to Ollama is lossless for our needs, converting up is not.
2. **Ollama via raw `httpx`**, not the `ollama` Python package — we need direct access to `logprobs`, `think`, `prompt_eval_count`, `eval_count`, and `done_reason` (truncation detection) without waiting on wrapper support.
3. **Telemetry has two tables**: `calls` (one row per model call — exactly what you specified) plus `routes` (one row per `complete()`, holds the final decision and the counterfactual). Counterfactual is stored on the route so it's counted once, not once per attempt.
4. **Counterfactual caching**: for local routes we assume the *observed* cloud cache-read ratio for that session (falls back to 0). Without this, agent loops (which resend a growing prefix) would massively overstate savings. Configurable: `counterfactual.cache_assumption: observed | none`.
5. **Pure pre-route** = tier picked from score only, no validation-based escalation. Structural errors (timeout/exception) still fall back to Sonnet, behind `pre_route.fallback_on_error: true`, because a crashed call isn't a routing result.
6. **Pure cascade** = always try local first, ignore the score for tier choice (score is still computed and logged), full validator chain.
7. **Dashboard** is a page of the web app (§10), not a separate server. It uses the same query layer as `report`. Charts use Recharts, bundled with the frontend so nothing is loaded from a CDN (works offline on the tailnet).
8. **CLI** = `typer` + `rich`.
9. **Scorer token count** uses a fast char-based estimator (no tokenizer call) so scoring stays in single-digit ms. Exact counts are recorded post-hoc from backend responses.
10. **Default cloud model IDs** in config: `anthropic/claude-sonnet-5.5` (sonnet tier), `anthropic/claude-opus-5.5` (opus tier), via OpenRouter. Change in `hybrid_router.yaml`.

## 2. Repository layout

```
orchestration-sdk/
├── pyproject.toml            # uv; extras: [learned] (sklearn, numpy), [dev]
├── CLAUDE.md
├── README.md
├── hybrid_router.example.yaml
├── pricing.yaml              # placeholder prices for the counterfactual; verify before trusting
├── src/hybrid_router/
│   ├── __init__.py           # HybridRouter, RouteDecision, load_config
│   ├── types.py              # Message, ContentBlock, ToolSpec, Usage, CompletionRequest/Response, StreamEvent
│   ├── config.py             # pydantic models for the YAML config + env overrides
│   ├── router.py             # HybridRouter: acomplete / astream / complete
│   ├── backends/
│   │   ├── base.py           # Backend ABC, BackendCapabilities, BackendError taxonomy
│   │   ├── registry.py       # kind -> class; entry-point group "hybrid_router.backends"
│   │   ├── ollama.py
│   │   └── openai_compat.py  # OpenRouter (cloud) and OpenAI-compatible local servers
│   ├── scoring/
│   │   ├── base.py           # Scorer protocol, ScoreResult(score, reasons, features)
│   │   ├── features.py       # feature extraction shared by all scorers
│   │   ├── rule.py           # RuleScorer
│   │   ├── embedding.py      # EmbeddingScorer (stub, phase 6)
│   │   └── learned.py        # LearnedScorer (stub, phase 6)
│   ├── zones.py              # score -> Zone(EASY|UNCERTAIN|HARD) + target tier
│   ├── session.py            # SessionState store: step index, stickiness countdown, cache ratio
│   ├── validators/
│   │   ├── base.py           # Validator ABC, ValidatorResult(ok, severity, reason, name, ms)
│   │   ├── structural.py
│   │   ├── heuristic.py
│   │   ├── confidence.py
│   │   ├── execution.py
│   │   ├── judge.py
│   │   └── chain.py          # ordered, short-circuits on first failure
│   ├── strategies/
│   │   ├── base.py           # Strategy ABC: run(request, ctx) -> (response, decision)
│   │   ├── hybrid.py
│   │   ├── pre_route.py
│   │   └── cascade.py
│   ├── telemetry/
│   │   ├── schema.sql
│   │   ├── store.py          # SQLite writer (WAL mode), migrations via user_version
│   │   └── queries.py        # shared by report, tune, dashboard
│   ├── pricing.py            # load pricing.yaml, cost(), counterfactual(), token ratio calibration
│   ├── reporting/
│   │   ├── report.py
│   │   ├── tune.py
│   │   └── whatif.py         # replay stored routes under a candidate config
│   ├── web/
│   │   ├── app.py            # FastAPI app factory
│   │   ├── auth.py           # bearer token dependency
│   │   ├── routes/           # chat.py, agents.py, config.py, compare.py, evals.py, stats.py
│   │   ├── runs.py           # background agent/compare/eval runs + SSE event bus
│   │   └── static/           # built frontend (generated, git-ignored, included in wheel)
│   ├── adapters/
│   │   ├── langgraph.py      # thin chat-model wrapper (optional import)
│   │   └── claude_agent_sdk.py  # delegate_local tool (see §9)
│   └── cli.py                # hybrid-router report|tune|web|eval|smoke|serve
├── frontend/                 # React + Vite + TS source (pnpm); `pnpm build` -> src/hybrid_router/web/static
├── examples/agent/           # example agent with 3 tools
├── evals/
│   ├── tasks.yaml            # ~30 labeled tasks
│   ├── checks.py
│   └── run_evals.py
└── tests/
    ├── fakes.py              # ScriptedBackend
    ├── unit/...
    └── integration/test_live.py
```

## 3. Core types (`types.py`)

- `Message(role: "user"|"assistant", content: str | list[ContentBlock])`, `system` passed separately.
- `ContentBlock` = `TextBlock | ToolUseBlock(id, name, input) | ToolResultBlock(tool_use_id, content, is_error) | ImageBlock`.
- `ToolSpec(name, description, input_schema: dict)`.
- `CompletionRequest(messages, system, tools, tool_choice, response_schema: dict|None, max_tokens, temperature, task_type, session_id, metadata)`.
- `Usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)`.
- `CompletionResponse(content: list[ContentBlock], stop_reason: end_turn|max_tokens|tool_use|error, usage, model, tier, latency_ms, token_logprobs: list[float]|None, raw)`.
- `RouteDecision(route_id, strategy, score, zone, reasons, features, tiers_attempted, attempts: list[AttemptRecord], final_tier, forced: bool, escalation_reason)`.
- `RoutedResponse(response, decision)` — what `acomplete` returns.

## 4. Phase 1 — Backends + smoke test

**Interface** (`backends/base.py`):

```python
class BackendCapabilities(BaseModel):
    tools: bool; streaming: bool; logprobs: bool; json_mode: bool
    prompt_caching: bool; exact_token_count: bool

class Backend(ABC):
    name: str; tier: Tier; model: str
    capabilities: BackendCapabilities
    async def generate(self, req: CompletionRequest, *, logprobs=False) -> CompletionResponse
    async def stream(self, req, *, logprobs=False) -> AsyncIterator[StreamEvent]
    async def count_tokens(self, req) -> int        # estimate (no exact counter for OpenRouter)
    async def health(self) -> bool
    async def aclose(self) -> None
```
Tool calling is part of `generate`/`stream` (tools on the request, `ToolUseBlock`s in the response) rather than a separate method. Errors are normalized into `BackendTimeout | BackendUnavailable | BackendBadResponse | BackendRateLimited`.

**OllamaBackend**: `/api/chat` with `tools`, `format` (JSON schema when `response_schema` set), `options.num_ctx`, `think`, `logprobs` (feature-detected via `/api/version`; capability is false on older Ollama and the confidence validator then skips). Converts Anthropic-shaped messages/tool results ↔ Ollama's. Truncation from `done_reason == "length"`. Per-call timeout from config (local can hang on a 16 GB machine under memory pressure).

**OpenAICompatBackend** (`kind: openai_compat`): raw `httpx2` against `/chat/completions` with SSE streaming. OpenRouter is the default target.
- Converts our messages to OpenAI format: tool results become `role: tool`, and images become data URLs.
- Reads `usage.cost` as the reported USD cost, and cached tokens from `prompt_tokens_details`.
- Keeps OpenRouter `reasoning_details` (Claude thinking) and replays them on later turns. Claude needs this for tool use with reasoning.
- For Anthropic models, adds `cache_control` breakpoints on the system prompt and the latest message.
- Claude 5.5 rejects `temperature` and forced `tool_choice`, so the Claude tiers turn both off and the backend records a warning when it drops them.
- `health()` checks that the configured model id exists in `/models`.
- With `logprobs: true` and no API key, the same backend drives llama.cpp, LM Studio, vLLM or MLX servers.

**Extensibility**: backends are constructed from config by `kind` via the registry; third parties register through the `hybrid_router.backends` entry-point group. The router only ever sees `Backend` + `tier`, so llama.cpp / MLX / NPU backends need zero router changes. Multiple backends per tier allowed later (ordered fallback).

**Config sketch** (`hybrid_router.yaml`):
```yaml
tiers:
  local:  {kind: ollama,    model: qwen3:8b, base_url: http://localhost:11434, timeout_s: 60, think: false}
  sonnet: {kind: openai_compat, model: anthropic/claude-sonnet-5.5, sampling_params: false, forced_tool_choice: false}
  opus:   {kind: openai_compat, model: anthropic/claude-opus-5.5, sampling_params: false, forced_tool_choice: false}
```

**Deliverables**: types, config, both backends, `hybrid-router smoke` (pings each tier, one plain call + one tool call, prints latency/tokens), unit tests for message/tool conversion with recorded fixtures.

## 5. Phase 2 — Scoring, zones, validators, strategies

### RuleScorer
Each feature is normalized to [0,1], then `score = clamp(Σ wᵢ·fᵢ / Σ wᵢ + stickiness_boost)`. Reasons list contains each feature that contributed above a threshold, e.g. `"tokens=9.4k (0.62)"`, `"keyword:/refactor|prove/"`.

| Feature | Normalization |
|---|---|
| context tokens | piecewise linear: 0 at `≤ low` (2k), 1 at `≥ high` (24k — near what an 8B model handles well) |
| tool count | `min(n / tools_saturation, 1)` |
| task_type | lookup table in config (`classify: 0.05, extract: 0.1, summarize: 0.15, chat: 0.3, code: 0.6, code_review: 0.8, plan: 0.85, multi_step_reasoning: 0.9`); unknown → `task_type_default` |
| hard keywords | regex list with per-pattern weight, on latest user text only |
| schema complexity | depth, property count, enums, `anyOf/oneOf`, required count → combined 0..1 |
| prior tool error | last tool_result in context has `is_error` on a step served locally → bump (see §5 execution note) |
| stickiness | additive boost `sticky_boost` (0.25) for next `sticky_steps` (3) steps after an escalation in the same session; decays linearly |

Target: < 2 ms per call; a unit test asserts it on a 30k-token context.

### Zones (`zones.py`)
`score < easy_threshold` (default 0.25) → EASY → local + structural validator only, streamed directly.
`score > hard_threshold` (0.65) → HARD → Sonnet, or Opus if `score > opus_threshold` (0.85).
Otherwise UNCERTAIN → cascade. Config validation enforces `easy ≤ hard ≤ opus`.

### Validator chain
Ordered cheapest-first, short-circuit on first failure. `severity ∈ {soft, hard}`.

1. **structural** (hard unless noted): exception/timeout, empty output, `max_tokens` truncation (soft), invalid JSON when schema requested, schema mismatch (`jsonschema`), tool call with unknown name, tool input failing that tool's `input_schema`.
2. **heuristic** (soft): refusal/uncertainty phrase list, n-gram repetition ratio, output shorter than `min_chars_by_task_type`, required fields empty.
3. **confidence** (soft): mean token logprob < threshold (skipped if backend lacks logprobs); optional two-sample disagreement (second local sample at higher temperature, compares normalized text / tool calls; low weight, off by default).
4. **execution** (hard on failure): for code outputs, `ast.parse`/`compile`; optional caller hook `test_runner(code) -> bool`. For tool calls the router can't run the tool — see note below.
5. **judge** (off by default): configurable model (local or Claude) asked for PASS/FAIL; counted in cost.

> **Note on "tool returned no error":** the tool runs *after* the router returns, so this can't gate the current step. Instead: when the next step's messages contain an erroring `tool_result` for a call made by a local step, (a) it raises the next step's score (feature above), and (b) telemetry back-fills `local_succeeded = false` on that previous call — the label `LearnedScorer` trains on. Flag if you want something stronger (e.g. an optional caller-supplied tool dry-run hook).

### Escalation rule
On validator failure: go to **Opus** if severity is `hard` *and* `escalate_hard_to_opus: true`, or if the original score > `opus_threshold`; otherwise **Sonnet**. If Sonnet itself fails structural validation → Opus once. Max attempts configurable (default 3).
Escalation payload: original request (default) or, with `escalation.mode: revise`, the original request plus the local draft and the failure reason as a "revise this" user turn.

### Strategies
- **hybrid**: score → zone → EASY / HARD direct, UNCERTAIN cascade.
- **pre_route**: score → tier, no validation-based escalation.
- **cascade**: always local first → full chain → escalate.
- `force_tier=` bypasses scoring decisions (score still computed and logged; `forced=true`).

### Streaming
`astream()` yields `StreamEvent`s. EASY zone and HARD-zone cloud calls stream live. UNCERTAIN: local output is buffered, validated, then either replayed as events or discarded and the escalated call is streamed live. A trailing `RouteDecision` event closes every stream.

### Session state
In-memory `SessionStore` (step counter, stickiness countdown, observed cache ratio), keyed by `session_id`, LRU-bounded. Interface allows a SQLite-backed store later for multi-process agents.

**Deliverables**: all of the above + unit tests with `ScriptedBackend` (returns pre-programmed responses/errors/latencies; zero network): per-feature scoring, zone boundaries, each validator pass/fail case, escalation paths (soft→Sonnet, hard→Opus, high-score→Opus, Sonnet-fails→Opus, revise mode), stickiness decay, force_tier, buffered vs live streaming.

## 6. Phase 3 — Telemetry, pricing, cost

**`routes`**: route_id, ts, session_id, step_index, task_type, strategy, score, zone, reasons(JSON), features(JSON), forced, final_tier, final_model, n_attempts, success, total_latency_ms, actual_cost_usd, counterfactual_cost_usd, cf_input_tokens_est, cf_output_tokens_est.

**`calls`**: call_id, route_id, ts, session_id, step_index, task_type, strategy, score, zone, reasons, tier, model, role (`direct | cascade_attempt | escalation`), attempt_index, validator_results(JSON), input/output/cache_read/cache_write tokens, latency_ms, ttft_ms, success, error, wasted (bool: a cascade attempt superseded by an escalation), local_succeeded (nullable, back-filled), actual_cost_usd, cost_source (`reported | pricing | local`), counterfactual_cost_usd (= route counterfactual on the final call, 0 on others so `SUM` is correct).

Indexes on ts, session_id, route_id. WAL mode; writes happen off the hot path via a background task with a bounded queue (flushed on `aclose`). Telemetry can be disabled.

**`pricing.yaml`**:
```yaml
# PLACEHOLDER PRICES for the counterfactual — verify each at https://openrouter.ai/models before trusting reports.
# Actual cost comes from OpenRouter's per-call usage.cost; these are only the fallback.
currency: USD
models:
  anthropic/claude-sonnet-5.5: {input_per_mtok: 0.0, output_per_mtok: 0.0, cache_read_per_mtok: 0.0, cache_write_per_mtok: 0.0}
  anthropic/claude-opus-5.5:   {input_per_mtok: 0.0, output_per_mtok: 0.0, cache_read_per_mtok: 0.0, cache_write_per_mtok: 0.0}
local:
  default_cost_usd: 0.0
  per_hour_usd: 0.0     # optional electricity/amortization; cost = per_hour_usd * latency_s / 3600
```
**Actual cost** = the backend's `reported_cost_usd` (OpenRouter `usage.cost`) when present (`cost_source=reported`), else priced from `pricing.yaml` (`cost_source=pricing`); local calls use the local settings (`cost_source=local`). The calibration of pricing.yaml against reported cost is shown in `report` so drift is visible.
Unknown model → cost recorded as NULL with a warning (never silently $0). Reports show "prices unverified" if all values are still 0.

**Counterfactual** (per route, baseline model from config):
- Final call was cloud at the baseline model → counterfactual = that call's actual cost.
- Final call was cloud on another model (e.g. Opus) → same tokens repriced at baseline (so Opus routes show *negative* savings — honest).
- Final call was local → Claude tokens ≈ local tokens × calibrated ratio; cache split per §1.4.
- Ratio calibration: median of `claude_input / local_input` over escalated routes (same request, both counts known), recomputed at report time and cached in a `meta` table.

**Cascade waste** = latency, tokens, and local cost of `calls.wasted = 1`.

**Deliverables**: store + migrations, pricing loader, cost/counterfactual functions, tests on hand-computed examples and an in-memory DB.

## 7. Phase 4 — Eval harness

- `evals/tasks.yaml`: ~30 tasks (≈10 easy / 12 medium / 8 hard) across classify, extract-to-schema, summarize, tool-use (single and multi-step with fake tools), code-gen with tests, reasoning/planning, code review.
- Each task: `id, difficulty, task_type, messages, tools?, response_schema?, check: {type: exact|regex|json_schema|json_fields|contains_all|python_tests|tool_call, ...}, min_tier_label` (hand label: the cheapest tier expected to pass).
- `run_evals.py --strategies hybrid,pre_route,cascade,all_local,all_cloud --repeat 1 --budget-usd 2.00`: writes to a separate `evals.sqlite` (same schema, tagged by run_id) and prints a comparison table:
  - quality: pass rate overall and by difficulty
  - routing accuracy vs hand label and vs **empirical label** (from the all_local run: did local pass?) — confusion matrix: under-escalated (local failed, sent local) vs over-escalated (sent to cloud, local would have passed)
  - cost actual/counterfactual, latency p50/p95, cascade waste
- Hard budget cap aborts cloud calls once exceeded.

## 8. Phase 5 — CLI report and tune

**`hybrid-router report --since 7d [--db path] [--json]`**: % calls and % tokens local vs cloud; actual vs counterfactual cost and savings (abs and %); cost and latency by zone; escalation rate by task_type; top escalation reasons (validator name + reason); cascade waste; p50/p95 latency per tier; token ratio used and sample count.

**`hybrid-router tune --since 30d`**:
- Bin cascade-zone routes by score (0.05 bins); escalation rate per bin.
- Suggest `hard_threshold` = lowest score above which escalation rate ≥ `target_escalation` (default 0.7) holds monotonically; `easy_threshold` = highest score below which local success ≥ 0.9.
- For each suggestion, replay telemetry: est. latency saved (waste removed by skipping local in newly-HARD bins), cost delta (cloud cost added for routes moved to cloud, incl. ones that would have succeeded locally), quality risk (local failures moved into EASY).
- Weight suggestions: logistic regression on stored features → escalated (uses sklearn if `[learned]` installed, else a small numpy-free gradient fit), reported as normalized weights next to current ones.
- Prints a suggested YAML patch; never edits config itself. Refuses to suggest with < 50 routes in a zone.
- `tune` returns structured suggestions (pydantic models), so the web app's tuning page renders and applies the same output.

**What-if replay** (`reporting/whatif.py`, shared by `tune` and the web app): every route stores its feature vector, so a candidate config (thresholds or weights) can re-score and re-zone past routes without any model calls. The output is the new zone mix, the routes that would change tier, and estimated cost, latency and quality-risk deltas. Each estimate is labelled with its basis: observed when the route actually ran at that tier, estimated otherwise.

## 9. Phase 6 — Example agent, adapters, scorer stubs

- `examples/agent/`: a "repo assistant" agent loop with 3 tools (`read_file`, `search_text`, `run_python` in a subprocess with timeout), using `HybridRouter` as its model provider with `session_id` per run, tagging steps with `task_type`.
- **Adapters**:
  - LangGraph: `HybridRouterChatModel(BaseChatModel)` in `adapters/langgraph.py`, optional import, converts LangChain messages ↔ ours, exposes `RouteDecision` in `response_metadata`.
  - Claude Agent SDK (supported pattern): an in-process MCP tool `delegate_local(task, task_type)` that sends simple subtasks through the router. Claude stays the main loop.
  - Proxy (`hybrid-router serve`) is deferred to an experimental phase 9: full SSE streaming with `ping` keep-alives while buffering, byte-exact pass-through for cloud, `x-claude-code-*` hint headers for session/request-class routing.
- `EmbeddingScorer` stub: interface, embedding via Ollama `/api/embed`, in-memory kNN over stored route embeddings (new `embeddings` table), score = fraction of k neighbors escalated; marked experimental.
- `LearnedScorer` stub: trains on `features` + `local_succeeded`, persists with joblib, `hybrid-router train-scorer`.
- README (written in phase 8, drafted earlier as phases land): `ollama pull qwen3:8b`, `ollama pull nomic-embed-text`, `uv sync`, `OPENROUTER_API_KEY`, config walkthrough, mermaid diagram (request → scorer → zone → strategy → backends → validators → telemetry → report/tune/dashboard), tuning guide.

## 10. Phase 7–8 — Orchestration web app

`hybrid-router web [--host 127.0.0.1] [--port 8765] [--fake-backends]` starts a single FastAPI process. It serves the API and the built React app, and it owns one shared `HybridRouter`, so all telemetry lands in the same SQLite DB as SDK and CLI usage.

### Pages
1. **Chat playground**
   - A streaming chat that goes through the router. You can pick the strategy, `task_type` and `force_tier`, and toggle the example tools.
   - Each assistant turn has a route card showing tier badge, score with a per-feature breakdown, zone, reasons, attempts (including discarded local drafts, expandable), validator results, tokens, latency/TTFT, actual vs counterfactual cost.
   - Conversations are persisted, and each one is one router `session_id`, so stickiness is visible as you chat.
2. **Agent runs**
   - Start the example agent (or any agent registered in config) on a task.
   - A live step timeline shows each step's tier, score, tool calls and results, escalations, and the running cost against the counterfactual.
   - Runs can be cancelled, and past runs can be replayed from telemetry.
3. **Compare**
   - Send one prompt (or one eval task) under several strategies (hybrid / pre_route / cascade / all_local / all_cloud) and see the outputs side by side with cost, latency and the eval check's pass/fail.
   - From the same page you can launch full eval runs, which show progress and then the comparison table from §7.
4. **Tuning**
   - Sliders for the thresholds and inputs for feature weights, stickiness and model per tier.
   - What-if replay against the last N days updates as you drag.
   - `tune` suggestions appear as one-click "apply to draft" changes.
   - Applying a draft validates it, writes `hybrid_router.yaml` (the previous version is kept as a timestamped backup and the change is shown as a diff first), and hot-swaps the router's config. Requests already in flight finish on the old config.
5. **Dashboard**: every `report` view as charts, with a time-range picker.
   - Local vs cloud split over time.
   - Cumulative cost vs counterfactual.
   - Cost and latency by zone.
   - Escalation rate by task_type.
   - Score histogram coloured by outcome.
   - p50/p95 latency per tier, and cascade waste.
   - Top escalation reasons.
   - A header strip shows Ollama/OpenRouter health and the loaded models.

### API sketch
- `POST /api/chat/{conversation_id}/messages`: SSE stream of `StreamEvent`s, ending with a `route_decision` event. Also `GET/POST/DELETE /api/conversations`.
- `POST /api/runs`: start an agent, compare or eval run, returns `run_id`.
  - `GET /api/runs/{id}/events` streams the run's events over SSE.
  - `POST /api/runs/{id}/cancel`.
- `GET /api/config`, `POST /api/config/validate`, `POST /api/config/whatif`, `PUT /api/config` (apply).
- `GET /api/tune/suggestions`; `GET /api/stats/*` (the report queries); `GET /api/health`.
- OpenAPI types are generated into the frontend (`openapi-typescript`) so the TypeScript and pydantic models can't drift.

### New storage
The same SQLite DB gets three new tables:
- `conversations` and `conversation_messages` (chat history).
- `runs`: kind, status, config snapshot hash, and summary. Agent steps link to `routes` via `session_id` = run_id.

Every route also records `config_hash`, so the dashboard can split metrics before and after a tuning change.

### Concurrency
A 16 GB Mac can serve only one local generation well at a time. Local backends therefore sit behind a configurable semaphore (default 1), and queued local calls show a "waiting for local model" state in the UI. Queue wait is recorded separately from model latency so it doesn't skew routing stats. Compare runs execute the local legs one after another and the cloud legs in parallel.

### Security (LAN / tailnet)
- The server binds `127.0.0.1` by default. A non-loopback `--host` refuses to start unless `HYBRID_ROUTER_WEB_TOKEN` is set. The SPA asks for the token once and keeps it in localStorage.
- No CORS. The frontend is same-origin.
- `OPENROUTER_API_KEY` stays server-side and is never returned by `/api/config`.
- The example agent's `run_python` tool is remote code execution if exposed. It is disabled by default in the web app, needs `agent.allow_code_exec: true`, and runs in a subprocess with a timeout, a temp working directory and no inherited env.
- Spend guard: a configurable daily cloud budget, plus a per-run cap for compare and eval runs. When a budget is hit, cloud tiers are blocked and the UI says so.
- The README recommends `tailscale serve` for HTTPS rather than building TLS into the app.

### Frontend stack
- React 18, Vite and TypeScript, with TanStack Query for data fetching.
- Recharts for charts, and Tailwind for styling.
- A small SSE helper with auto-reconnect handles streaming.
- pnpm for development only. The wheel ships the built assets, so end users need no Node.

### Testing
- API: pytest + httpx `AsyncClient` against the app wired to `ScriptedBackend`s. This covers SSE event order, auth enforcement, config apply/rollback, and budget blocking.
- Frontend: Vitest for components such as the route card and what-if math display.
- End to end: one Playwright test against `hybrid-router web --fake-backends`, using the pre-installed Chromium. It covers sending a chat message, seeing an escalation route card, and moving a threshold slider to watch the what-if update.

## 11. Engineering conventions (→ CLAUDE.md)

Python 3.11+, uv, src layout, pydantic v2 for all config and data models, `ruff` (lint + format), `pyright` basic, pytest + `pytest-asyncio`. No network in unit tests (`ScriptedBackend`). Integration tests in `tests/integration/`, auto-skipped unless Ollama responds at the configured URL **and** `OPENROUTER_API_KEY` is set. No hardcoded model names or prices outside example config / pricing.yaml. Router never imports a concrete backend. Every routing decision is explainable via `reasons`. Frontend: TypeScript strict, API types generated from OpenAPI (never hand-written), `pnpm lint && pnpm test` must pass; built assets are not committed. Each phase ends with a stop for review.

## 12. Risks / things to watch

- `qwen3:8b` tool-calling reliability is mediocre on multi-tool tasks; expect higher escalation in agent steps until thresholds are tuned. The validators (unknown tool, schema-invalid input) are the main guard.
- Logprobs support depends on your Ollama version; confidence validation silently degrades to skipped (logged).
- Memory pressure on 16 GB (model + IDE + browser) can cause slow first tokens; local timeout and `keep_alive` are configurable, and timeouts escalate as hard failures.
- Web app scope roughly doubles the project; the SDK stays fully usable without it (`web` is an optional extra: `uv sync --extra web`), and phases 1–6 don't depend on it.
- Hot-swapping config mid-session changes routing for in-flight agent runs on their *next* step; runs record the config hash per step so this is visible.
- Counterfactual is an estimate; reports label it as such and show the token ratio and sample size.

## 13. Build order and review gates

1. Backends + smoke test → **stop**
2. Scoring, zones, validators, strategies + unit tests → **stop**
3. Telemetry, pricing, cost/counterfactual → **stop**
4. Eval harness → **stop**
5. CLI report/tune + what-if replay → **stop**
6. Example agent, adapters, Embedding/Learned scorer stubs → **stop**
7. Web app, part 1: FastAPI API (auth, SSE, conversations, runs, config apply, stats), frontend scaffold, **Chat playground** and **Dashboard** pages → **stop**
8. Web app, part 2: **Agent runs**, **Compare + eval launcher** and **Tuning** pages, Playwright e2e test, final README (setup, web app, mermaid diagram, tuning guide) → **stop**

(CLAUDE.md is created in phase 1 and updated each phase.)
