#!/usr/bin/env bash
# One-time setup: checks prerequisites, installs Python and frontend dependencies,
# builds the web UI. Safe to re-run. Usage: ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; exit 1; }

echo "Checking prerequisites"

command -v uv >/dev/null 2>&1 \
  || fail "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh   (then open a new terminal)"
ok "uv $(uv --version | awk '{print $2}')"

command -v node >/dev/null 2>&1 \
  || fail "Node.js not found. Install Node 22 LTS:  brew install node@22   (or from https://nodejs.org)"
NODE_VERSION=$(node -p 'process.versions.node')
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=12)||(a===20&&b>=19)?0:1)' \
  || fail "Node $NODE_VERSION is too old; need 20.19+ or 22.12+:  brew install node@22"
ok "node $NODE_VERSION"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "  pnpm not found; enabling it via corepack (ships with Node)"
  corepack enable pnpm || fail "couldn't enable pnpm. Try:  npm install -g pnpm@10"
fi
ok "pnpm $(cd frontend && pnpm --version)"

if command -v ollama >/dev/null 2>&1; then
  ok "ollama $(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
else
  warn "ollama not found (needed for the local model, not for --fake-backends): brew install ollama"
fi

echo
echo "Installing Python dependencies (uv downloads Python 3.12 if you don't have it)"
uv sync --extra web
ok "Python environment ready in .venv"

echo
echo "Installing and building the web UI"
(cd frontend && pnpm install --frozen-lockfile && pnpm build)
ok "web UI built"

echo
echo "Done. Next:"
echo "  uv run hybrid-router web --fake-backends     # try the app with demo models (no Ollama/key needed)"
echo "  uv run hybrid-router smoke                   # check Ollama + OpenRouter once they're set up"
echo "  uv run hybrid-router web                     # the real thing: http://127.0.0.1:8765"
