# Perception-XAlpha two-stage stock selector

Status: **research-only / shadow-only / not a trading signal**.

## Why this exists

The V5 factor miner evaluates a broad linear rank tilt.  Taking four successful
train-screen factors and concentrating them into ten stocks changes the portfolio being
tested.  A direct replay showed that the frozen four-factor top ten lost money in the
latest historical quarantine.  This module makes the research objective match the desired
decision more closely without changing any trading code.

## Frozen design

Stage one calculates four point-in-time factor percentiles and equally averages them:

1. downside-volatility asymmetry;
2. book-to-price minus 120-session momentum;
3. point-in-time growth composite times 120-session momentum;
4. multi-period reversal.

The first stage only supplies the best 50 securities each day.  It does not buy them.

Stage two uses one `HistGradientBoostingClassifier` and one
`HistGradientBoostingRegressor`.  Their 21 inputs contain the four factor ranks, factor
agreement, past stock returns/volatility/liquidity/bar shape, and past market
return/volatility/breadth.  The label is the return from the next session's open through
the open ten sessions later.

The base models are fitted on the early training dates.  The next ten trading dates are
purged.  The final 126 training dates calibrate probability with isotonic regression and
expected return with a strongly regularised one-variable ridge map.  Validation and
shadow labels never fit or calibrate a model.

The frozen selection policy requires both:

- calibrated probability of a positive ten-day return of at least 55%; and
- calibrated expected ten-day return of at least 0.30%.

At most ten securities are retained.  Fewer than ten, including zero, is a valid result.

## Features deliberately skipped

- Historical industry-relative strength: the current industry map is incomplete and not
  point-in-time.
- Historical announcement/CSRC/audit risk: no point-in-time event ledger currently covers
  the full sample.
- Analyst revisions: no point-in-time consensus history is available.

These are data gaps, not values to approximate from current information.

## Evaluation

The study reports:

- Brier score, LogLoss, AUC and ECE versus a constant training prior;
- probability-bucket hit rates;
- number of selected observations and average exposure;
- ten-day basket return, win rate and market-relative return;
- cumulative/annualised return, turnover and maximum drawdown;
- the identical metrics for the frozen four-factor top-ten baseline.

A model is not even considered a stable historical increment unless validation and shadow
both improve basket mean return and win rate and also improve probability quality over the
constant prior.  Passing this diagnostic still cannot promote the model.

## Known interpretation limits

The validation and shadow dates were viewed in earlier diagnostics, so they are not a
pristine final holdout.  Current-master survivorship, incomplete historical ST status and
raw unadjusted prices also remain.  Results therefore describe a historical diagnostic,
not a reliable forecast of future profit.

## Running

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_perception_xalpha_two_stage.py
```

Artifacts are written under:

```text
outputs/edge_research/perception_xalpha_two_stage_selector/<run_id>/
```

Every artifact is marked research-only; `orders` is always empty.  The script does not
read or write live trading configuration, strategy overlays, positions, broker state,
`build_decision()`, or execution locks.
