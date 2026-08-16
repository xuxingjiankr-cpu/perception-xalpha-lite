"""Render docs/selection_artifact.svg from the selection-artifact example.

The figure in the README is generated, not drawn: it is produced by the same code path that
prints the numbers, so the picture cannot drift away from the result it illustrates. Re-run
after changing the example and commit the output.

Run: python examples/make_selection_artifact_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selection_artifact import (  # noqa: E402  (same directory, deliberate)
    SEED,
    SPLIT,
    TOP_K,
    annualised_ir,
    build_world,
    decile_excess,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "selection_artifact.svg"
WIDTH, HEIGHT = 880, 380
PAD_L, PAD_R, PAD_T, PAD_B = 62, 210, 54, 46
# Chosen to stay legible on GitHub in both light and dark themes.
INK = "#94a3b8"
HINDSIGHT = "#ef4444"
HONEST = "#3b82f6"


def curves() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(SEED)
    forward, factors = build_world(rng)
    excess = {name: decile_excess(signal, forward) for name, signal in factors.items()}
    trailing = {name: series.iloc[:SPLIT] for name, series in excess.items()}
    outcome = {name: series.iloc[SPLIT:] for name, series in excess.items()}
    by_hindsight = sorted(outcome, key=lambda name: -outcome[name].mean())[:TOP_K]
    by_trailing = sorted(trailing, key=lambda name: -annualised_ir(trailing[name]))[:TOP_K]
    hindsight = pd.concat([outcome[name] for name in by_hindsight], axis=1).mean(axis=1)
    honest = pd.concat([outcome[name] for name in by_trailing], axis=1).mean(axis=1)
    return (1.0 + hindsight).cumprod(), (1.0 + honest).cumprod()


def path_for(series: pd.Series, low: float, high: float) -> str:
    plot_width = WIDTH - PAD_L - PAD_R
    plot_height = HEIGHT - PAD_T - PAD_B
    span = max(high - low, 1e-9)
    points = []
    for index, value in enumerate(series.to_numpy()):
        x = PAD_L + plot_width * index / max(len(series) - 1, 1)
        y = PAD_T + plot_height * (1.0 - (value - low) / span)
        points.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(points)


def main() -> None:
    hindsight, honest = curves()
    low = min(hindsight.min(), honest.min(), 1.0)
    high = max(hindsight.max(), honest.max(), 1.0)
    baseline_y = PAD_T + (HEIGHT - PAD_T - PAD_B) * (1.0 - (1.0 - low) / max(high - low, 1e-9))
    end_h, end_o = float(hindsight.iloc[-1]), float(honest.iloc[-1])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">
  <title>Selection artifact: 400 pure-noise factors, true edge zero</title>
  <text x="{PAD_L}" y="26" font-size="16" font-weight="600" fill="{INK}">400 factors. All pure noise. True edge: exactly zero.</text>
  <text x="{PAD_L}" y="44" font-size="12.5" fill="{INK}" opacity="0.85">Same factors, same data, same window &#8212; only the rule for choosing the top {TOP_K} differs.</text>
  <line x1="{PAD_L}" y1="{baseline_y:.1f}" x2="{WIDTH - PAD_R}" y2="{baseline_y:.1f}" stroke="{INK}" stroke-opacity="0.35" stroke-dasharray="4 4"/>
  <text x="{PAD_L - 8}" y="{baseline_y + 4:.1f}" font-size="11" text-anchor="end" fill="{INK}" opacity="0.7">1.00</text>
  <path d="{path_for(hindsight, low, high)}" fill="none" stroke="{HINDSIGHT}" stroke-width="2.4"/>
  <path d="{path_for(honest, low, high)}" fill="none" stroke="{HONEST}" stroke-width="2.4"/>
  <g font-size="13">
    <rect x="{WIDTH - PAD_R + 6}" y="{PAD_T + 6}" width="11" height="11" fill="{HINDSIGHT}" rx="2"/>
    <text x="{WIDTH - PAD_R + 24}" y="{PAD_T + 16}" fill="{INK}" font-weight="600">chosen on the</text>
    <text x="{WIDTH - PAD_R + 24}" y="{PAD_T + 32}" fill="{INK}" font-weight="600">outcome window</text>
    <text x="{WIDTH - PAD_R + 24}" y="{PAD_T + 50}" fill="{HINDSIGHT}" font-size="12.5">IR {annualised_ir(hindsight.pct_change().dropna()):.2f} &#183; &#215;{end_h:.2f}</text>
    <rect x="{WIDTH - PAD_R + 6}" y="{PAD_T + 82}" width="11" height="11" fill="{HONEST}" rx="2"/>
    <text x="{WIDTH - PAD_R + 24}" y="{PAD_T + 92}" fill="{INK}" font-weight="600">chosen on</text>
    <text x="{WIDTH - PAD_R + 24}" y="{PAD_T + 108}" fill="{INK}" font-weight="600">trailing data only</text>
    <text x="{WIDTH - PAD_R + 24}" y="{PAD_T + 126}" fill="{HONEST}" font-size="12.5">IR {annualised_ir(honest.pct_change().dropna()):.2f} &#183; &#215;{end_o:.2f}</text>
  </g>
  <text x="{PAD_L}" y="{HEIGHT - 16}" font-size="12" fill="{INK}" opacity="0.85">Neither line learned anything. The red one is what a backtest reports when the factor set is chosen after the fact.</text>
  <text x="{WIDTH - PAD_R + 6}" y="{HEIGHT - 16}" font-size="11" fill="{INK}" opacity="0.6">examples/selection_artifact.py</text>
</svg>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}  (hindsight x{end_h:.3f} vs honest x{end_o:.3f})")


if __name__ == "__main__":
    main()
