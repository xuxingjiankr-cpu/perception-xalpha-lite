"""Render docs/daily-rotation.svg — the live rotation record against the onshore indices.

Two rules govern this chart, and they are the reason it is worth publishing at all.

*The backtested span is drawn differently from the live span.* Everything before the
specification was frozen is an in-sample re-measurement of factors chosen from a 456-candidate
search, on data that overlaps the search. It is drawn dashed and greyed, behind a shaded panel
labelled as proving nothing. Only the live span is evidence, and at the start there is almost
none of it — which the chart shows honestly rather than hiding behind a long backtest.

*Like is compared with like.* The strategy line is a raw return net of realised cost, because
that is the only thing comparable to an index. The excess-over-universe line — the actual
research metric, which strips the market out — is drawn separately and labelled as a different
question. Plotting an excess against a raw index is the oldest trick in the genre.

Standard library only: this runs on a GitHub runner every day.

Run: python examples/render_rotation_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "docs" / "data" / "rotation.jsonl"
NEXT_PICK = ROOT / "docs" / "data" / "next_pick.json"
OUT = ROOT / "docs" / "daily-rotation.svg"

W, H = 1000, 520
PAD = {"l": 62, "r": 168, "t": 64, "b": 58}
PLOT_W, PLOT_H = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]

LINES = [
    ("strategy", "One name, rotated daily", "var(--accent)", 2.6),
    ("universe", "Eligible universe, equal weight", "var(--muted)", 1.6),
    ("sh_000300", "CSI300", "var(--line-a)", 1.6),
    ("sh_000016", "SSE50 (onshore A50 proxy)", "var(--line-b)", 1.6),
]


def read_rows() -> list[dict]:
    if not SERIES.exists():
        raise SystemExit(f"no series at {SERIES}")
    rows = []
    for line in SERIES.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return sorted(rows, key=lambda row: row["date"])


def cumulative(rows: list[dict]) -> dict[str, list[float | None]]:
    """Compound each series into an index starting at 0%, carrying gaps forward."""
    series: dict[str, list[float | None]] = {key: [] for key, *_ in LINES}
    series["excess"] = []
    running = {key: 1.0 for key in series}
    for row in rows:
        daily = {
            # Raw return of the held name after realised turnover cost: comparable to an index.
            "strategy": float(row["strategy_gross"]) - float(row["cost"]),
            "universe": row.get("universe"),
            "sh_000300": row.get("sh_000300"),
            "sh_000016": row.get("sh_000016"),
            # The research metric: the same book with the market taken out.
            "excess": float(row["strategy_net"]),
        }
        for key, value in daily.items():
            if value is not None:
                running[key] *= 1.0 + float(value)
            series[key].append(running[key] - 1.0)
    return series


def path_for(values: list[float | None], xs: list[float], lo: float, hi: float, span: range) -> str:
    scale = PLOT_H / (hi - lo) if hi > lo else 0.0
    points = []
    for index in span:
        value = values[index]
        if value is None:
            continue
        y = PAD["t"] + PLOT_H - (value - lo) * scale
        points.append(f"{'M' if not points else 'L'}{xs[index]:.1f},{y:.1f}")
    return " ".join(points)


def read_next_pick() -> dict | None:
    if not NEXT_PICK.exists():
        return None
    try:
        return json.loads(NEXT_PICK.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    rows = read_rows()
    series = cumulative(rows)
    count = len(rows)
    live_start = next((index for index, row in enumerate(rows) if row.get("phase") == "live"), count)

    plotted = [value for key, *_ in LINES for value in series[key] if value is not None]
    plotted += [value for value in series["excess"] if value is not None]
    lo, hi = min(plotted + [0.0]), max(plotted + [0.0])
    margin = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - margin, hi + margin
    xs = [PAD["l"] + (PLOT_W * index / max(1, count - 1)) for index in range(count)]

    def y_of(value: float) -> float:
        return PAD["t"] + PLOT_H - (value - lo) * (PLOT_H / (hi - lo))

    parts: list[str] = []
    step = max(0.05, round((hi - lo) / 5 / 0.05) * 0.05)
    tick = (int(lo / step)) * step
    while tick <= hi:
        if lo <= tick <= hi:
            y = y_of(tick)
            parts.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" x2="{PAD["l"]+PLOT_W}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{PAD["l"]-10}" y="{y+4:.1f}" text-anchor="end">{tick*100:+.0f}%</text>')
        tick += step

    if live_start > 0:
        edge = xs[min(live_start, count - 1)]
        parts.append(f'<rect class="backtest" x="{PAD["l"]}" y="{PAD["t"]}" width="{edge-PAD["l"]:.1f}" height="{PLOT_H}"/>')
        parts.append(f'<text class="phase" x="{PAD["l"]+10}" y="{PAD["t"]+20}">backtest — in-sample, not evidence</text>')
    if live_start < count:
        edge = xs[live_start]
        parts.append(f'<line class="divider" x1="{edge:.1f}" y1="{PAD["t"]}" x2="{edge:.1f}" y2="{PAD["t"]+PLOT_H}"/>')
        parts.append(f'<text class="phase live" x="{edge+10:.1f}" y="{PAD["t"]+20}">live — frozen spec, forward only</text>')
    else:
        parts.append(f'<text class="phase live" x="{PAD["l"]+PLOT_W-8}" y="{PAD["t"]+20}" text-anchor="end">live record starts here</text>')

    back, live = range(0, min(live_start + 1, count)), range(live_start, count)
    for key, label, colour, width in LINES:
        if live_start > 0:
            d = path_for(series[key], xs, lo, hi, back)
            if d:
                parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
                             f'stroke-dasharray="4 4" opacity="0.5"/>')
        if live_start < count:
            d = path_for(series[key], xs, lo, hi, live)
            if d:
                parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"/>')

    d = path_for(series["excess"], xs, lo, hi, range(count))
    if d:
        parts.append(f'<path d="{d}" fill="none" stroke="var(--excess)" stroke-width="1.4" stroke-dasharray="1 3"/>')

    legend_y = PAD["t"] + 6
    for key, label, colour, _ in LINES:
        final = next((value for value in reversed(series[key]) if value is not None), 0.0)
        parts.append(f'<line x1="{PAD["l"]+PLOT_W+16}" y1="{legend_y-4}" x2="{PAD["l"]+PLOT_W+36}" y2="{legend_y-4}" stroke="{colour}" stroke-width="2.6"/>')
        parts.append(f'<text class="legend" x="{PAD["l"]+PLOT_W+42}" y="{legend_y}">{label}</text>')
        parts.append(f'<text class="legend value" x="{PAD["l"]+PLOT_W+42}" y="{legend_y+16}">{final*100:+.1f}%</text>')
        legend_y += 44
    final_excess = next((value for value in reversed(series["excess"]) if value is not None), 0.0)
    parts.append(f'<line x1="{PAD["l"]+PLOT_W+16}" y1="{legend_y-4}" x2="{PAD["l"]+PLOT_W+36}" y2="{legend_y-4}" stroke="var(--excess)" stroke-width="1.4" stroke-dasharray="1 3"/>')
    parts.append(f'<text class="legend" x="{PAD["l"]+PLOT_W+42}" y="{legend_y}">excess over universe</text>')
    parts.append(f'<text class="legend value" x="{PAD["l"]+PLOT_W+42}" y="{legend_y+16}">{final_excess*100:+.1f}%</text>')

    for index in (0, count - 1):
        parts.append(f'<text class="tick" x="{xs[index]:.1f}" y="{PAD["t"]+PLOT_H+22}" '
                     f'text-anchor="{"start" if index == 0 else "end"}">{rows[index]["date"]}</text>')

    # The forward pick, stated on the chart itself. Publishing the name before the session it
    # applies to is what makes the record a prediction rather than a report; putting it only in
    # a data file would leave the shared image saying nothing that could later be wrong.
    pick = read_next_pick()
    if pick:
        box_w, box_x, box_y = PAD["r"] - 24, PAD["l"] + PLOT_W + 16, H - 118
        parts.append(f'<rect class="pick-box" x="{box_x}" y="{box_y}" width="{box_w}" height="72" rx="8"/>')
        parts.append(f'<text class="pick-label" x="{box_x+12}" y="{box_y+20}">NEXT SESSION</text>')
        parts.append(f'<text class="pick-symbol" x="{box_x+12}" y="{box_y+44}">{pick["symbol"]}</text>')
        parts.append(f'<text class="pick-meta" x="{box_x+12}" y="{box_y+62}">published {pick["as_of"]} · buy at open</text>')

    live_count = count - live_start
    footer = (
        f"{live_count} live session{'s' if live_count != 1 else ''} of a frozen one-name book"
        f" · {count - live_start if live_count else count} backtested"
        f" · spec {rows[-1].get('spec_sha256', '')}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img"
     aria-label="Cumulative return of a one-name daily rotation against onshore A-share indices">
  <style>
    :root {{ --bg:#ffffff; --ink:#1f2328; --dim:#656d76; --grid:#e6e8eb; --muted:#8b949e;
             --accent:#7c3aed; --line-a:#0969da; --line-b:#bf8700; --excess:#57606a;
             --shade:rgba(101,109,118,0.07); }}
    @media (prefers-color-scheme: dark) {{
      :root {{ --bg:#0d1117; --ink:#e6edf3; --dim:#8b949e; --grid:#21262d; --muted:#6e7681;
               --accent:#a371f7; --line-a:#58a6ff; --line-b:#d29922; --excess:#8b949e;
               --shade:rgba(139,148,158,0.10); }}
    }}
    :root[data-theme="dark"] {{ --bg:#0d1117; --ink:#e6edf3; --dim:#8b949e; --grid:#21262d;
      --muted:#6e7681; --accent:#a371f7; --line-a:#58a6ff; --line-b:#d29922; --excess:#8b949e;
      --shade:rgba(139,148,158,0.10); }}
    :root[data-theme="light"] {{ --bg:#ffffff; --ink:#1f2328; --dim:#656d76; --grid:#e6e8eb;
      --muted:#8b949e; --accent:#7c3aed; --line-a:#0969da; --line-b:#bf8700; --excess:#57606a;
      --shade:rgba(101,109,118,0.07); }}
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .title {{ font-size: 19px; font-weight: 600; fill: var(--ink); }}
    .subtitle, .footer {{ font-size: 12.5px; fill: var(--dim); }}
    .tick {{ font-size: 11px; fill: var(--dim); }}
    .legend {{ font-size: 12px; fill: var(--ink); }}
    .legend.value {{ font-size: 13px; font-weight: 600; fill: var(--dim); }}
    .phase {{ font-size: 11px; fill: var(--dim); }}
    .phase.live {{ fill: var(--accent); font-weight: 600; }}
    .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .backtest {{ fill: var(--shade); }}
    .divider {{ stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 3 3; }}
    .pick-box {{ fill: none; stroke: var(--accent); stroke-width: 1.2; }}
    .pick-label {{ font-size: 9.5px; letter-spacing: 0.09em; fill: var(--accent); font-weight: 700; }}
    .pick-symbol {{ font-size: 19px; font-weight: 700; fill: var(--ink);
                    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .pick-meta {{ font-size: 10px; fill: var(--dim); }}
  </style>
  <rect width="{W}" height="{H}" fill="var(--bg)"/>
  <text class="title" x="{PAD['l']}" y="30">Hold one name, rotate every session</text>
  <text class="subtitle" x="{PAD['l']}" y="49">Cumulative return after 30 bps per rotation, against the indices it is competing with</text>
  {chr(10) + '  '.join(parts)}
  <text class="footer" x="{PAD['l']}" y="{H-16}">{footer}</text>
  <text class="footer" x="{W-PAD['r']+16}" y="{H-16}" text-anchor="start">Research only. No orders.</text>
</svg>
"""
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} — {count} sessions, {live_count} live, spec {rows[-1].get('spec_sha256','')}")


if __name__ == "__main__":
    main()
