import { useEffect, useState } from "react";

export type ThemeChoice = "system" | "light" | "dark";
const KEY = "hybrid-router-theme";

export function loadTheme(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
  try {
    if (choice === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, choice);
  } catch {
    /* per-viewer convenience only */
  }
  window.dispatchEvent(new Event("hr-theme"));
}

const TOKENS = [
  "surface",
  "ink",
  "ink-2",
  "muted",
  "grid",
  "axis",
  "tier-local",
  "tier-sonnet",
  "tier-opus",
  "outcome-escalated",
  "ord-light",
  "ord-dark",
] as const;
export type ChartTokens = Record<(typeof TOKENS)[number], string>;

function read(): ChartTokens {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(
    TOKENS.map((t) => [t, style.getPropertyValue(`--${t}`).trim() || "#888"]),
  ) as ChartTokens;
}

/** Resolved CSS tokens for SVG charts; re-read when the theme changes. */
export function useChartTokens(): ChartTokens {
  const [tokens, setTokens] = useState<ChartTokens>(read);
  useEffect(() => {
    const update = () => setTokens(read());
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", update);
    window.addEventListener("hr-theme", update);
    return () => {
      mq.removeEventListener("change", update);
      window.removeEventListener("hr-theme", update);
    };
  }, []);
  return tokens;
}
