import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api, getToken, setToken } from "../api/client";
import type { Health } from "../api/types";
import { usd } from "../lib/format";
import { applyTheme, loadTheme, type ThemeChoice } from "../lib/theme";
import { Button, Status, cx } from "./ui";

const NAV = [
  { to: "/chat", label: "Chat" },
  { to: "/agent", label: "Agent runs" },
  { to: "/compare", label: "Compare" },
  { to: "/tuning", label: "Tuning" },
  { to: "/dashboard", label: "Dashboard" },
];

function HealthStrip() {
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: () => api<Health>("/api/health"),
    refetchInterval: 30_000,
  });
  if (!data) return null;
  const b = data.budget;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1" data-testid="health">
      {data.fake && (
        <span className="rounded-full border border-line px-2 py-0.5 text-xs text-ink-2">demo models</span>
      )}
      {data.tiers.map((t) => (
        <span key={t.tier} title={`${t.model}: ${t.detail}`}>
          <Status ok={t.ok} label={`${t.tier} · ${t.model}`} />
        </span>
      ))}
      {b.daily_cloud_budget_usd != null && (
        <span className="text-xs text-ink-2" title="Today's cloud spend vs daily budget">
          {b.cloud_allowed ? "" : "⚠ "}cloud today {usd(b.spent_today_usd ?? 0, 2)} / {usd(b.daily_cloud_budget_usd, 2)}
        </span>
      )}
    </div>
  );
}

function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>(loadTheme);
  useEffect(() => applyTheme(choice), [choice]);
  const next: Record<ThemeChoice, ThemeChoice> = { system: "light", light: "dark", dark: "system" };
  return (
    <Button variant="ghost" className="px-2 text-xs" onClick={() => setChoice(next[choice])} title="Theme">
      {choice === "system" ? "◐ Auto" : choice === "light" ? "☀ Light" : "☾ Dark"}
    </Button>
  );
}

function TokenGate() {
  const [needed, setNeeded] = useState(false);
  const [value, setValue] = useState("");
  useEffect(() => {
    const on = () => setNeeded(true);
    window.addEventListener("hr-unauthorized", on);
    return () => window.removeEventListener("hr-unauthorized", on);
  }, []);
  if (!needed) return null;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    setToken(value.trim());
    setNeeded(false);
    window.location.reload();
  };
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl border border-line bg-surface p-5">
        <h2 className="text-base font-semibold">Access token</h2>
        <p className="mt-1 text-sm text-ink-2">
          This server requires the token from <code>HYBRID_ROUTER_WEB_TOKEN</code>.
          {getToken() && " The saved token was rejected."}
        </p>
        <input
          autoFocus
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="mt-3 w-full rounded-md border border-line bg-surface px-3 py-2 text-sm"
          aria-label="Token"
        />
        <Button variant="primary" type="submit" className="mt-3 w-full">
          Continue
        </Button>
      </form>
    </div>
  );
}

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2">
          <span className="font-semibold">hybrid-router</span>
          <nav className="flex flex-wrap gap-1" aria-label="Main">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  cx(
                    "rounded-md px-2.5 py-1 text-sm",
                    isActive ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:text-ink",
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <HealthStrip />
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-4">
        <Outlet />
      </main>
      <TokenGate />
    </div>
  );
}
