"""``hybrid-router report``: local vs cloud split, cost vs counterfactual, zones,
escalations, cascade waste and latency. ``build_report`` returns plain data so the
web dashboard renders exactly the same numbers."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from hybrid_router.pricing import Pricing
from hybrid_router.reporting.queries import load_calls, load_routes, meta, pct
from hybrid_router.types import Usage


def _share(n: float, d: float) -> float | None:
    return n / d if d else None


def _normalize_reason(reason: str) -> str:
    """Group 'local failed heuristic: suspiciously short (12 < 40 chars)' variants."""
    reason = re.sub(r"\(.*?\)", "", reason)
    reason = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", reason)
    reason = re.sub(r"\d+(\.\d+)?", "N", reason)
    return re.sub(r"\s+", " ", reason).strip(" :")


def build_report(
    conn: sqlite3.Connection, since_ts: float, pricing: Pricing | None = None
) -> dict[str, Any]:
    routes = load_routes(conn, since_ts)
    calls = load_calls(conn, since_ts)
    n = len(routes)

    local_routes = [r for r in routes if r.final_tier == "local"]
    tokens = defaultdict(int)
    for c in calls:
        side = "local" if c["tier"] == "local" else "cloud"
        tokens[side] += (
            (c["input_tokens"] or 0)
            + (c["output_tokens"] or 0)
            + (c["cache_read_tokens"] or 0)
            + (c["cache_write_tokens"] or 0)
        )
    actual = sum(r.actual_cost_usd or 0 for r in routes)
    cf = sum(r.counterfactual_cost_usd or 0 for r in routes)
    unknown_cost = sum(1 for r in routes if r.actual_cost_usd is None)

    overview = {
        "routes": n,
        "calls": len(calls),
        "local_route_share": _share(len(local_routes), n),
        "local_token_share": _share(tokens["local"], tokens["local"] + tokens["cloud"]),
        "tokens_local": tokens["local"],
        "tokens_cloud": tokens["cloud"],
        "actual_cost_usd": actual,
        "counterfactual_cost_usd": cf,
        "savings_usd": cf - actual,
        "savings_pct": _share(cf - actual, cf),
        "routes_with_unknown_cost": unknown_cost,
        "escalation_rate": _share(sum(r.escalated for r in routes), n),
        "success_rate": _share(sum(r.success for r in routes), n),
        "prices_verified": pricing.all_verified if pricing else None,
        "token_ratio_samples": len(meta(conn, "token_ratio_samples") or []),
    }

    by_zone = []
    for zone in ("easy", "uncertain", "hard"):
        zr = [r for r in routes if r.zone == zone]
        if not zr:
            continue
        lat = [r.total_latency_ms for r in zr]
        by_zone.append(
            {
                "zone": zone,
                "routes": len(zr),
                "local_share": _share(sum(r.final_tier == "local" for r in zr), len(zr)),
                "escalation_rate": _share(sum(r.escalated for r in zr), len(zr)),
                "cost_usd": sum(r.actual_cost_usd or 0 for r in zr),
                "avg_cost_usd": sum(r.actual_cost_usd or 0 for r in zr) / len(zr),
                "counterfactual_usd": sum(r.counterfactual_cost_usd or 0 for r in zr),
                "latency_p50_ms": pct(lat, 50),
                "latency_p95_ms": pct(lat, 95),
            }
        )

    by_task: dict[str, list] = defaultdict(list)
    for r in routes:
        by_task[r.task_type or "(none)"].append(r)
    by_task_type = sorted(
        (
            {
                "task_type": t,
                "routes": len(rs),
                "escalation_rate": _share(sum(r.escalated for r in rs), len(rs)),
                "local_share": _share(sum(r.final_tier == "local" for r in rs), len(rs)),
                "avg_score": sum(r.score for r in rs) / len(rs),
            }
            for t, rs in by_task.items()
        ),
        key=lambda x: -(x["escalation_rate"] or 0),
    )

    reasons = Counter(
        _normalize_reason(r.escalation_reason)
        for r in routes
        if r.escalated and r.escalation_reason
    )
    validators = Counter(r.local_failed_validator for r in routes if r.local_failed_validator)

    wasted = [c for c in calls if c["wasted"]]
    local_calls = [c for c in calls if c["tier"] == "local"]
    waste = {
        "wasted_calls": len(wasted),
        "local_calls": len(local_calls),
        "wasted_share_of_local": _share(len(wasted), len(local_calls)),
        "latency_ms": sum(c["latency_ms"] or 0 for c in wasted),
        "tokens": sum((c["input_tokens"] or 0) + (c["output_tokens"] or 0) for c in wasted),
        "cost_usd": sum(c["actual_cost_usd"] or 0 for c in wasted),
    }

    latency = []
    for tier in ("local", "sonnet", "opus"):
        tc = [c for c in calls if c["tier"] == tier]
        if not tc:
            continue
        latency.append(
            {
                "tier": tier,
                "calls": len(tc),
                "p50_ms": pct([c["latency_ms"] for c in tc], 50),
                "p95_ms": pct([c["latency_ms"] for c in tc], 95),
                "ttft_p50_ms": pct([c["ttft_ms"] for c in tc if c["ttft_ms"] is not None], 50),
                "queue_p95_ms": pct([c["queue_ms"] or 0 for c in tc], 95),
            }
        )

    # Pricing drift: what pricing.yaml would have charged vs what the provider reported.
    drift = None
    if pricing:
        est = rep = 0.0
        for c in calls:
            if c["cost_source"] != "reported" or c["tier"] == "local":
                continue
            price = pricing.lookup(c["model"] or "")
            if price is None:
                continue
            est += price.cost(
                Usage(
                    input_tokens=c["input_tokens"] or 0,
                    output_tokens=c["output_tokens"] or 0,
                    cache_read_tokens=c["cache_read_tokens"] or 0,
                    cache_write_tokens=c["cache_write_tokens"] or 0,
                )
            )
            rep += c["actual_cost_usd"] or 0
        drift = {"reported_usd": rep, "pricing_yaml_usd": est, "ratio": _share(est, rep)}

    # Time series and score histogram (dashboard charts).
    span = (routes[-1].ts - routes[0].ts) if n > 1 else 0
    bucket = 3600 if span <= 3 * 86400 else 86400
    series: dict[int, dict[str, float]] = {}
    for r in routes:
        b = int(r.ts // bucket * bucket)
        s = series.setdefault(b, {"ts": b, "local": 0, "cloud": 0, "actual": 0.0, "cf": 0.0})
        s["local" if r.final_tier == "local" else "cloud"] += 1
        s["actual"] += r.actual_cost_usd or 0
        s["cf"] += r.counterfactual_cost_usd or 0
    hist = []
    for i in range(20):
        lo, hi = i / 20, (i + 1) / 20
        br = [r for r in routes if lo <= r.score < hi or (i == 19 and r.score == 1.0)]
        hist.append(
            {
                "lo": lo,
                "hi": hi,
                "local_ok": sum(r.final_tier == "local" and not r.escalated for r in br),
                "escalated": sum(r.escalated for r in br),
                "cloud_direct": sum(
                    r.final_tier not in (None, "local") and not r.escalated for r in br
                ),
            }
        )

    return {
        "since_ts": since_ts,
        "overview": overview,
        "by_zone": by_zone,
        "by_task_type": by_task_type,
        "top_escalation_reasons": [{"reason": k, "count": v} for k, v in reasons.most_common(10)],
        "failing_validators": [{"validator": k, "count": v} for k, v in validators.most_common()],
        "cascade_waste": waste,
        "latency_by_tier": latency,
        "pricing_drift": drift,
        "timeseries": {"bucket_s": bucket, "points": [series[k] for k in sorted(series)]},
        "score_histogram": hist,
    }


def _p(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def _ms(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.2f}s" if v >= 1000 else f"{v:.0f}ms"


def _usd(v: float | None) -> str:
    return "-" if v is None else f"${v:,.4f}"


def render_report(rep: dict[str, Any], console: Console) -> None:
    o = rep["overview"]
    if not o["routes"]:
        console.print("No routes recorded in this period.")
        return
    console.print(
        f"[bold]{o['routes']} routes[/], {o['calls']} model calls  ·  "
        f"local: {_p(o['local_route_share'])} of routes, "
        f"{_p(o['local_token_share'])} of tokens  ·  "
        f"escalation rate {_p(o['escalation_rate'])}"
    )
    console.print(
        f"cost {_usd(o['actual_cost_usd'])} vs all-cloud {_usd(o['counterfactual_cost_usd'])}  →  "
        f"[green]saved {_usd(o['savings_usd'])} ({_p(o['savings_pct'])})[/]"
    )
    if o["prices_verified"] is False:
        console.print(
            "[yellow]pricing.yaml prices are unverified: the all-cloud figure is an "
            "estimate until you check them and set verified: true[/]"
        )
    if o["routes_with_unknown_cost"]:
        console.print(
            f"[yellow]{o['routes_with_unknown_cost']} routes used a model with no "
            "price and no reported cost[/]"
        )

    t = Table(title="By zone")
    for c in (
        "zone",
        "routes",
        "local",
        "escalated",
        "cost",
        "avg cost",
        "all-cloud",
        "p50",
        "p95",
    ):
        t.add_column(c)
    for z in rep["by_zone"]:
        t.add_row(
            z["zone"],
            str(z["routes"]),
            _p(z["local_share"]),
            _p(z["escalation_rate"]),
            _usd(z["cost_usd"]),
            _usd(z["avg_cost_usd"]),
            _usd(z["counterfactual_usd"]),
            _ms(z["latency_p50_ms"]),
            _ms(z["latency_p95_ms"]),
        )
    console.print(t)

    t = Table(title="Escalation rate by task_type")
    for c in ("task_type", "routes", "escalated", "served local", "avg score"):
        t.add_column(c)
    for x in rep["by_task_type"]:
        t.add_row(
            x["task_type"],
            str(x["routes"]),
            _p(x["escalation_rate"]),
            _p(x["local_share"]),
            f"{x['avg_score']:.2f}",
        )
    console.print(t)

    if rep["top_escalation_reasons"]:
        t = Table(title="Top escalation reasons")
        t.add_column("count")
        t.add_column("reason")
        for x in rep["top_escalation_reasons"]:
            t.add_row(str(x["count"]), x["reason"])
        console.print(t)

    w = rep["cascade_waste"]
    console.print(
        f"Cascade waste: {w['wasted_calls']} of {w['local_calls']} local calls discarded "
        f"({_p(w['wasted_share_of_local'])}), {_ms(w['latency_ms'])} and {w['tokens']:,} tokens"
    )

    t = Table(title="Latency by tier")
    for c in ("tier", "calls", "p50", "p95", "ttft p50", "queue p95"):
        t.add_column(c)
    for x in rep["latency_by_tier"]:
        t.add_row(
            x["tier"],
            str(x["calls"]),
            _ms(x["p50_ms"]),
            _ms(x["p95_ms"]),
            _ms(x["ttft_p50_ms"]),
            _ms(x["queue_p95_ms"]),
        )
    console.print(t)

    d = rep["pricing_drift"]
    if d and d["reported_usd"]:
        console.print(
            f"pricing.yaml vs reported cost: {d['ratio']:.2f}x "
            f"({_usd(d['pricing_yaml_usd'])} vs {_usd(d['reported_usd'])})"
        )


def report_json(rep: dict[str, Any]) -> str:
    return json.dumps(rep, indent=2, default=str)
