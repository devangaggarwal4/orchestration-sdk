# hybrid-router

Run as much of an agent's LLM work as possible on a local model (Ollama), and escalate to Claude (Sonnet, then Opus) only when needed. Every call records telemetry, actual cost, and what it would have cost to send everything to the cloud.

> **Status: phase 1 of 8.** The backend layer works today: Ollama and Claude backends, config loading and a smoke test. The router, telemetry, reports and web app follow. See [`docs/PLAN.md`](docs/PLAN.md).

## Setup (macOS)

```bash
# 1. Local model (16 GB Mac)
brew install ollama && ollama serve          # or the Ollama.app
ollama pull qwen3:8b

# 2. Python env (Python 3.11+)
uv sync

# 3. Claude
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Check every tier
uv run hybrid-router smoke                   # --skip-cloud for local only
```

`smoke` runs three checks on each tier:

- A health check.
- A streamed one-word reply, reporting time to first token, total latency, tokens and logprobs.
- One tool call.

The cloud tiers make a few tiny API calls.

## Configuration

Copy `hybrid_router.example.yaml` to `hybrid_router.yaml` and edit it.

- **Load order:** built-in defaults, then the YAML file, then environment overrides. The file is `./hybrid_router.yaml`, or the path in `$HYBRID_ROUTER_CONFIG` or `--config`.
- **Environment overrides** look like `HYBRID_ROUTER__TIERS__LOCAL__MODEL=llama3.1:8b`.
- **See the result:** `uv run hybrid-router config` prints the resolved config.

## Using a backend directly

```python
import asyncio
from hybrid_router import CompletionRequest, Message, Tier, load_config
from hybrid_router.backends import create_backend

async def main():
    cfg = load_config()
    async with create_backend(cfg.tiers[Tier.LOCAL], tier=Tier.LOCAL) as local:
        resp = await local.generate(
            CompletionRequest(messages=[Message(role="user", content="Hi!")]),
            logprobs=True,
        )
        print(resp.text, resp.usage, resp.mean_logprob)

asyncio.run(main())
```

## Development

```bash
uv run pytest -q                 # unit tests (no network)
uv run pytest -m integration     # live; needs Ollama + ANTHROPIC_API_KEY
uv run ruff check src tests && uv run pyright
```
