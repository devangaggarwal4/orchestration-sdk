import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The built app is served by the Python package (`hybrid-router web`).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../src/hybrid_router/web/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
