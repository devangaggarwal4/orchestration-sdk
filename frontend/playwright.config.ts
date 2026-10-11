import { defineConfig } from "@playwright/test";

// Boots the real server with demo models in a throwaway directory, against the
// built frontend (run `pnpm build` first).
const PORT = 8799;

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  webServer: {
    command: `sh -c 'cd "$(mktemp -d)" && uv run --project "${process.cwd()}/.." hybrid-router web --fake-backends --port ${PORT}'`,
    url: `http://127.0.0.1:${PORT}/api/meta`,
    timeout: 60_000,
    reuseExistingServer: false,
  },
});
