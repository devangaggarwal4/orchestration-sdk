#!/usr/bin/env bash
# Everything that must pass before a commit. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check src tests examples
uv run ruff check src tests examples
uv run pyright
uv run pytest -q
if [ -d frontend/node_modules ]; then
  (cd frontend && pnpm -s lint && pnpm -s typecheck && pnpm -s test)
fi
