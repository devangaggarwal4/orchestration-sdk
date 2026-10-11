import type { ButtonHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { TIER_LABEL } from "../lib/format";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Button({
  variant = "secondary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" }) {
  return (
    <button
      {...props}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium",
        "transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        variant === "primary" && "bg-accent text-white hover:brightness-110",
        variant === "secondary" && "border border-line bg-surface text-ink hover:bg-surface-2",
        variant === "ghost" && "text-ink-2 hover:bg-surface-2 hover:text-ink",
        className,
      )}
    />
  );
}

export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cx(
        "rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink",
        "focus-visible:outline-2 focus-visible:outline-accent",
        className,
      )}
    />
  );
}

export function Card({
  title,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("rounded-xl border border-line bg-surface p-4", className)}>
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      {children}
    </section>
  );
}

const TIER_CLASS: Record<string, string> = {
  local: "bg-local",
  sonnet: "bg-sonnet",
  opus: "bg-opus",
};

/** Tier identity: a colored dot beside ink text (text never wears the series color). */
export function TierBadge({ tier, className }: { tier: string | null | undefined; className?: string }) {
  if (!tier) return <span className="text-muted">–</span>;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border border-line px-2 py-0.5 text-xs font-medium text-ink",
        className,
      )}
    >
      <span className={cx("size-2 rounded-full", TIER_CLASS[tier] ?? "bg-muted")} aria-hidden />
      {TIER_LABEL[tier] ?? tier}
    </span>
  );
}

/** Status: icon + label + color, never color alone. */
export function Status({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-xs text-ink-2">
      <span aria-hidden className={ok ? "text-good" : "text-bad"}>
        {ok ? "✓" : "✕"}
      </span>
      {label}
    </span>
  );
}

export function Spinner({ label = "Working" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted" role="status">
      <span className="size-3 animate-spin rounded-full border-2 border-muted border-t-transparent" />
      {label}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-8 text-center text-sm text-muted">{children}</p>;
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p className="rounded-md border border-line bg-surface-2 px-3 py-2 text-sm text-ink" role="alert">
      <span className="text-bad" aria-hidden>
        ✕{" "}
      </span>
      {error instanceof Error ? error.message : String(error)}
    </p>
  );
}
