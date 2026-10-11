# hybrid-router

A Python SDK that sends LLM calls to a local model (Ollama) when it can and escalates to Claude (Sonnet, then Opus) through OpenRouter when it must. It records telemetry and cost for every call. The full design is in `docs/PLAN.md`. Work is built phase by phase, and the user reviews each phase before the next one starts.

## Commands

```bash
./scripts/setup.sh                       # first-time setup: checks tools, uv sync --extra web, builds the UI
uv sync --extra web                      # Python deps only (dev group included)
./scripts/check.sh                       # the commit gate: ruff format/lint, pyright, pytest, frontend lint/types/tests
uv run pytest -m integration             # live tests: needs Ollama + OPENROUTER_API_KEY
uv run hybrid-router smoke [--skip-cloud]
uv run hybrid-router config | report | tune | whatif | eval [--simulate] | train-scorer
uv run hybrid-router web [--fake-backends]
cd frontend && pnpm build && pnpm e2e    # Playwright against `web --fake-backends`
cd frontend && pnpm gen:api              # after changing any API/event model
```

`scripts/check.sh` must pass before a commit. Don't pipe it into `tail` without checking `PIPESTATUS`, because the pipe hides failures.

## Layout

- `src/hybrid_router/types.py`: the canonical types. Messages are Anthropic-shaped (role plus content blocks).
- `src/hybrid_router/backends/`:
  - `base.py`: the `Backend` ABC and the error taxonomy.
  - `registry.py`: maps a config `kind` to a backend class.
  - `ollama.py`: Ollama's `/api/chat`.
  - `openai_compat.py`: OpenAI-style `/chat/completions`. It serves OpenRouter for the cloud tiers, and llama.cpp, LM Studio, vLLM and MLX servers locally.
- `src/hybrid_router/config.py`: built-in defaults, then the YAML file, then `HYBRID_ROUTER__*` env overrides. It has sections for tiers, routing, scorer, validators, telemetry, agent and web.
- Routing:
  - `scoring/`: `rule.py` combines features noisy-OR; `learned.py` is logistic regression; `embedding.py` is a kNN stub.
  - `zones.py` maps scores to zones; `strategies.py` holds hybrid, pre_route and cascade.
  - `validators/`: structural, heuristic, confidence, execution and judge, plus the chain that runs them.
  - `router.py`: `HybridRouter`. `astream` is the single code path; `acomplete` and `complete` wrap it.
  - `decision.py`: `RouteDecision`, attempt records and router events.
  - `session.py`: stickiness and tool-error attribution.
- Telemetry and cost:
  - `telemetry/store.py`: SQLite (`routes`, `calls`, `meta`) with a background writer thread.
  - `pricing.py` and `pricing.yaml`: prices for the counterfactual and fallback.
  - `cost.py`: actual cost, counterfactual, and token-ratio calibration.
- Reporting:
  - `reporting/queries.py`: the shared read layer.
  - `report.py`, `tune.py` and `whatif.py`, used by both the CLI and the web app.
- `evals/`:
  - `tasks.yaml`: 30 labeled tasks, each with a reference answer.
  - `checks.py`, `tools.py`, `runner.py`, and `simulate.py` (offline tiers).
- `agent.py`: the example tool-using agent, which yields events. `adapters/` holds the LangGraph chat model and the Agent SDK `delegate_local` tool.
- `web/`:
  - `app.py`: the FastAPI app.
  - `runs.py`: background runs with SSE fan-out and replay.
  - `store.py`: conversations and runs.
  - `demo.py`: demo backends for `--fake-backends`.
  - `static/`: the built frontend. It's generated and git-ignored.
- `frontend/`: React, Vite, TypeScript, Tailwind v4, TanStack Query and Recharts. `src/api/schema.d.ts` is generated; never edit it by hand.
- `testing.py`: `ScriptedBackend` and `Reply`, used for unit tests.
- `tests/unit/`: no network. `tests/integration/`: live tests. `tests/fixtures/`: hand-written wire samples matching Ollama's `api/types.go` and OpenRouter's documented SSE, not live recordings.

## Conventions

- Python 3.11+ (3.12 pinned in `.python-version`), uv, src layout. Node 20.19+/22.12+ and pnpm 10 (`packageManager` in `frontend/package.json`) for the UI. The README's setup section is the user-facing install guide: keep it, `scripts/setup.sh` and its troubleshooting table in sync with the code's error messages. Use pydantic v2 for every config and data model, with `extra="forbid"` on configs.
- **Nothing outside `backends/` sees a wire format.** The router must never import a concrete backend; it gets one from `create_backend(config, tier=...)`.
- **New runtimes** (llama.cpp, MLX, NPU) get their own `Backend` and `BackendConfig` subclasses, registered with `@register_backend` or the `hybrid_router.backends` entry-point group. No router changes.
- **No hardcoded model names or prices** outside the config defaults in `config.py`, the per-backend config defaults, `hybrid_router.example.yaml` and `pricing.yaml`.
- **Backends raise only `BackendError` subclasses.** Each one has a `retryable` flag. Map SDK and transport errors at the backend boundary.
- **Every backend streams internally.** `generate()` consumes `stream()`. The last event is always `MessageEndEvent`.
- **Never use `asyncio.timeout` around a `yield` in an async generator.** It cancels the consumer's task. Use a deadline checked per chunk instead, as `OllamaBackend.stream` does.
- All HTTP goes through `httpx2`. Don't add `httpx` or provider SDKs: the user has no Anthropic API key, so the cloud goes through OpenRouter only.
- **Rules for Claude 5.5 models, which apply through OpenRouter too:**
  - No `temperature`, and no forced `tool_choice` (`any` or `tool`). The Claude tiers set `sampling_params: false` and `forced_tool_choice: false`, and the backend records a `warnings` entry when it drops either.
  - No assistant prefill.
  - Reasoning must be replayed unchanged. OpenRouter returns it as `reasoning_details`; we keep it in an `OpaqueBlock` and send it back on the assistant turn.
  - Keep histories append-only.
- **Cost:** OpenRouter's `usage.cost` lands in `CompletionResponse.reported_cost_usd` and is the actual cost. `pricing.yaml` only feeds the counterfactual and the fallback when no cost is reported.
- **Every routing decision must be explainable** through its `reasons`, starting in phase 2.
- **Tests:** unit tests must never touch the network. Use `ScriptedBackend`, a mock transport, or `httpx2.ASGITransport` for the web API.
- **Charts** use the dataviz skill's validated palette, keyed by entity: local is blue, Sonnet orange, Opus aqua, escalated violet. Every chart has a table view. A new color combination must be run through the palette validator first.
- **Web security:**
  - Binding anywhere but loopback requires `HYBRID_ROUTER_WEB_TOKEN`.
  - There is no CORS.
  - API keys never appear in `/api/config`. The config only holds env var *names*.
  - `run_python` stays off by default.
- The default for a 16 GB Mac is `qwen3:8b` with `think: false`.

## Decisions log

These were agreed with the user:

- **Cloud provider:** no Anthropic key, ever. Claude is reached through OpenRouter (`openai_compat`), and the `anthropic` backend and SDK were removed. Tiers stay local, then sonnet, then opus.
- The counterfactual cost baseline is configurable and defaults to the Sonnet tier.
- Local token counts are scaled to Claude tokens by a ratio calibrated from escalated calls.
- The API is async-first with a sync wrapper.
- Tool errors: when a local step's tool call errors, the next step's score goes up and `local_succeeded=false` is back-filled on that step.
- **Agent SDK:** our own agent loop and the LangGraph adapter come first. For the Claude Agent SDK we provide a `delegate_local` tool. The Anthropic-compatible proxy is a later, experimental extra.
- **Web app:** FastAPI with React, Vite and TypeScript. Single user, reachable over LAN or tailnet with a bearer token. Phases 7–8.
