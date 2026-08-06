# Perception-XAlpha Horizon Precision V3

## Research question

V2 found twelve train-only, prefix-causal daily factors, but its one-session combined
model did not predict ordinary loss well enough and failed the shadow gate.  V3 does not
mine or reweight those factors.  It tests whether two fixed changes improve the metric a
stock picker actually needs:

`P(open[t+1] -> executable exit is positive after 30 bps round-trip cost)`.

The fixed changes are:

1. compare 1, 2, 3, 5 and 10 trading-session holding periods; and
2. compare the frozen weighted factor composite with a majority-consensus filter that
   requires at least 6 of 12 factors to place the stock in their own top 20%.

Top 1, Top 3 and Top 10 are reported separately.  This is a declared grid of 30 policy
trials, not one unreported search followed by a best-looking result.

## Causal timing and execution

- Every factor uses data available through day-t close only.
- Entry is attempted at day-t+1 open and a sealed-up/untradeable entry is skipped.
- The intended exit is the open after the fixed holding period.
- If that exit is locked, the label carries the position to the first sellable open within
  five trading days.  It never silently replaces the locked exit with another stock.
- Each train/validation/shadow period drops enough dates from its own tail to contain the
  full holding-plus-five-day-exit window inside that period.  No outcome crosses a split.
- A future exit condition never changes the signal-time ranking.
- Round-trip cost is 0.30% for every resolved position.

Longer-horizon daily signals overlap.  V3 therefore reports day-basket precision and a
Newey-West t-statistic with `holding_days - 1` lags in addition to descriptive stock-level
precision.

## Preventing fake precision

The report always includes:

- signal-day coverage;
- mean names selected per signal day;
- intended, resolved and unresolved observations;
- stock-level gross and cost-positive win rates;
- day-basket cost-positive win rate and 95% Wilson lower bound;
- mean net return, severe-loss rate, CVaR and drawdown; and
- delayed-exit frequency.

This prevents a rule that selects almost nothing from claiming success merely through
abstention.  The development choice is made from the original train dates only.
Validation and shadow outcomes can reject the choice but cannot select another horizon,
factor, threshold or Top-N.

## Interpretation boundary

The factor book was itself discovered using the V2 training window, while validation and
shadow periods have already been read.  Consequently even a historical pass can only
create a separately preregistered fresh-forward hypothesis.  V3 cannot place an order,
change a position, write an overlay, modify `build_decision()`, change a risk gate, or
promote itself.

Run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_perception_xalpha_horizon_precision_v3.py
```

Artifacts are written under:

`outputs/edge_research/perception_xalpha_horizon_precision_v3/<run_id>/`

The latest candidate CSV is diagnostic only and is never an order or recommendation.
