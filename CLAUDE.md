# hybrid-router

A Python SDK that sends LLM calls to a local model (Ollama) when it can and escalates to Claude (Sonnet, then Opus) when it must. It records telemetry and cost for every call. The full design is in `docs/PLAN.md`. Work is built phase by phase, and the user reviews each phase before the next one starts.

## Commands

```bash
uv sync                                  # install (dev group included)
uv run pytest -q                         # unit tests; integration auto-skips
uv run pytest -m integration             # live tests: needs Ollama + ANTHROPIC_API_KEY
uv run ruff format src tests && uv run ruff check src tests
uv run pyright                           # must be 0 errors
uv run hybrid-router smoke [--skip-cloud]
uv run hybrid-router config              # print resolved config
```

All four of format, lint, pyright and tests must be clean before a commit.

## Layout

- `src/hybrid_router/types.py`: the canonical types. Messages are Anthropic-shaped (role plus content blocks).
- `src/hybrid_router/backends/`:
  - `base.py`: the `Backend` ABC and the error taxonomy.
  - `registry.py`: maps a config `kind` to a backend class.
  - `ollama.py` and `anthropic.py`: the two implementations.
- `src/hybrid_router/config.py`: built-in defaults, then the YAML file, then `HYBRID_ROUTER__*` env overrides.
- `src/hybrid_router/cli.py`: the `hybrid-router` command, built with typer.
- `tests/unit/`: no network. Backends run against `httpx2.MockTransport`.
- `tests/integration/`: live tests, skipped automatically when Ollama or the API key is missing.
- `tests/fixtures/`: wire-format samples. They are hand-written to match Ollama's `api/types.go` and Anthropic's SSE events, not recorded from live calls.

## Conventions

- Python 3.11+, uv, src layout. Use pydantic v2 for every config and data model, with `extra="forbid"` on configs.
- **Nothing outside `backends/` sees a wire format.** The router must never import a concrete backend; it gets one from `create_backend(config, tier=...)`.
- **New runtimes** (llama.cpp, MLX, NPU) get their own `Backend` and `BackendConfig` subclasses, registered with `@register_backend` or the `hybrid_router.backends` entry-point group. No router changes.
- **No hardcoded model names or prices** outside the config defaults in `config.py`, the per-backend config defaults, `hybrid_router.example.yaml` and `pricing.yaml`.
- **Backends raise only `BackendError` subclasses.** Each one has a `retryable` flag. Map SDK and transport errors at the backend boundary.
- **Every backend streams internally.** `generate()` consumes `stream()`. The last event is always `MessageEndEvent`.
- **Never use `asyncio.timeout` around a `yield` in an async generator.** It cancels the consumer's task. Use a deadline checked per chunk instead, as `OllamaBackend.stream` does.
- HTTP goes through `httpx2`, the same stack `anthropic` 1.x uses. Don't add `httpx`.
- **Rules for current Claude models (Sonnet 5.5 / Opus 5.5):**
  - No `temperature`.
  - No forced `tool_choice` (`any` or `tool`).
  - No assistant prefill.
  - Thinking blocks must be replayed unchanged.

  `AnthropicBackend` drops or downgrades these and records a `warnings` entry. Keep histories append-only.
- **Every routing decision must be explainable** through its `reasons`, starting in phase 2.
- **Tests:** unit tests must never touch the network. Use `ScriptedBackend` (phase 2+) or a mock transport.
- The default for a 16 GB Mac is `qwen3:8b` with `think: false`.

## Decisions log

These were agreed with the user:

- The counterfactual cost baseline is configurable and defaults to the Sonnet tier.
- Local token counts are scaled to Claude tokens by a ratio calibrated from escalated calls.
- The API is async-first with a sync wrapper.
- Tool errors: when a local step's tool call errors, the next step's score goes up and `local_succeeded=false` is back-filled on that step.
- **Agent SDK:** our own agent loop and the LangGraph adapter come first. For the Claude Agent SDK we provide a `delegate_local` tool. The Anthropic-compatible proxy is a later, experimental extra.
- **Web app:** FastAPI with React, Vite and TypeScript. Single user, reachable over LAN or tailnet with a bearer token. Phases 7–8.
