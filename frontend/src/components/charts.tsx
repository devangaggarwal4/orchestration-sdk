import { useState, type ReactNode } from "react";

interface TipProps {
  active?: boolean;
  label?: unknown;
  payload?: readonly { dataKey?: unknown; value?: unknown; name?: unknown; color?: string }[];
}
import { Button, Card } from "./ui";

export interface Column<T> {
  key: string;
  label: string;
  value: (row: T) => ReactNode;
  numeric?: boolean;
}

/** Chart card with a table-view twin (every value reachable without hovering). */
export function ChartCard<T>({
  title,
  subtitle,
  rows,
  columns,
  children,
  legend,
  dimmed,
}: {
  title: string;
  subtitle?: string;
  rows: T[];
  columns: Column<T>[];
  children: ReactNode;
  legend?: { label: string; color: string; shape?: "rect" | "line" }[];
  dimmed?: boolean;
}) {
  const [table, setTable] = useState(false);
  return (
    <Card
      title={
        <span>
          {title}
          {subtitle && <span className="ml-2 font-normal text-muted">{subtitle}</span>}
        </span>
      }
      actions={
        <Button variant="ghost" className="px-2 py-1 text-xs" onClick={() => setTable((t) => !t)}>
          {table ? "Chart" : "Table"}
        </Button>
      }
      className={dimmed ? "opacity-60 transition-opacity" : "transition-opacity"}
    >
      {legend && legend.length > 1 && !table && (
        <ul className="mb-2 flex flex-wrap gap-3 text-xs text-ink-2">
          {legend.map((l) => (
            <li key={l.label} className="flex items-center gap-1.5">
              <span
                aria-hidden
                style={{ background: l.color }}
                className={l.shape === "line" ? "h-0.5 w-3 rounded" : "size-2.5 rounded-sm"}
              />
              {l.label}
            </li>
          ))}
        </ul>
      )}
      {table ? <DataTable rows={rows} columns={columns} /> : children}
    </Card>
  );
}

export function DataTable<T>({ rows, columns }: { rows: T[]; columns: Column<T>[] }) {
  if (!rows.length) return <p className="py-6 text-center text-sm text-muted">No data</p>;
  return (
    <div className="max-h-72 overflow-auto">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-surface text-muted">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "py-1 pr-3 text-right" : "py-1 pr-3"}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tnum text-ink">
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "py-1 pr-3 text-right" : "py-1 pr-3"}>
                  {c.value(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Tooltip: values lead, series names follow, keyed by a short line of the series color. */
export function makeTooltip(format: (v: number, name: string) => string, labelFormat?: (l: unknown) => string) {
  return function ChartTooltip({ active, payload, label }: TipProps) {
    if (!active || !payload?.length) return null;
    return (
      <div className="rounded-md border border-line bg-surface px-2.5 py-2 text-xs shadow-sm">
        <div className="mb-1 text-muted">{labelFormat ? labelFormat(label) : String(label)}</div>
        {payload.map((p) => (
          <div key={String(p.dataKey)} className="flex items-center gap-2">
            <span aria-hidden className="h-0.5 w-3 rounded" style={{ background: p.color }} />
            <span className="tnum font-semibold text-ink">{format(Number(p.value), String(p.name))}</span>
            <span className="text-ink-2">{String(p.name)}</span>
          </div>
        ))}
      </div>
    );
  };
}

export function StatTile({
  label,
  value,
  detail,
  hero,
  children,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  hero?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4" data-testid={`stat-${label}`}>
      <div className="text-xs text-ink-2">{label}</div>
      <div className={hero ? "mt-1 text-5xl font-semibold text-ink" : "mt-1 text-2xl font-semibold text-ink"}>
        {value}
      </div>
      {detail && <div className="mt-1 text-xs text-muted">{detail}</div>}
      {children}
    </div>
  );
}

/** Budget meter: fill carries severity, track is a lighter step of the same ramp. */
export function Meter({ value, max }: { value: number; max: number }) {
  const frac = Math.min(value / max, 1);
  const color = frac >= 1 ? "var(--status-critical)" : frac >= 0.8 ? "var(--status-warning)" : "var(--accent)";
  return (
    <div className="mt-2 h-1.5 w-full rounded-full bg-surface-2" role="meter" aria-valuenow={value} aria-valuemax={max}>
      <div className="h-full rounded-full" style={{ width: `${frac * 100}%`, background: color }} />
    </div>
  );
}
