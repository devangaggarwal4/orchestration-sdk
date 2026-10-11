export function usd(v: number | null | undefined, digits?: number): string {
  if (v == null) return "–";
  const d = digits ?? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 4);
  const s = Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
  return `${v < 0 ? "−" : ""}$${s}`;
}

/** Axis ticks for money: keep two significant digits for sub-dollar values. */
export function usdTick(v: number): string {
  if (v === 0) return "$0";
  if (Math.abs(v) >= 1) return usd(v, Math.abs(v) >= 100 ? 0 : 2);
  return `$${Number(v.toPrecision(2))}`;
}

export function pct(v: number | null | undefined, digits = 0): string {
  return v == null ? "–" : `${(v * 100).toFixed(digits)}%`;
}

export function ms(v: number | null | undefined): string {
  if (v == null) return "–";
  if (Math.abs(v) >= 60_000) return `${(v / 60_000).toFixed(1)}m`;
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(v >= 10_000 ? 0 : 1)}s`;
  return `${Math.round(v)}ms`;
}

export function compact(v: number | null | undefined): string {
  if (v == null) return "–";
  return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

export function signed(v: number, fmt: (x: number) => string): string {
  return `${v > 0 ? "+" : v < 0 ? "−" : "±"}${fmt(Math.abs(v))}`;
}

export function timeAgo(ts: number): string {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export const TIER_LABEL: Record<string, string> = {
  local: "Local",
  sonnet: "Sonnet",
  opus: "Opus",
};
