# Perception-XAlpha win-rate V2

Status: **research-only / shadow-only / not a trading signal**.

## Diagnosis

The first two-stage selector failed because its absolute ten-day win classifier did not
survive a market base-rate shift.  The candidate positive-return rate was 55.1% in the
validation block but only 35.9% in shadow.  The frozen model continued to predict roughly
48.7%, its shadow AUC fell below 0.5, and its highest probability bucket performed worse
than the candidate pool.

Raising the probability threshold cannot repair an inverted ranking.  It only reduces
the number of trades.

## Preregistered solution

V2 separates stock selection from long-only market timing.

1. The same four frozen factors create a point-in-time top-50 candidate pool.
2. Stock-specific inputs are converted to same-date cross-sectional percentiles.
3. A ridge model predicts each candidate's relative ten-day return percentile rather
   than its absolute return.
4. The model is refitted every 21 trading days on at most the latest 756 trading days.
5. Every fit/test boundary purges ten complete trading days.
6. The five highest predicted ranks are eligible only when information already known at
   the signal close shows both:
   - median-market 20-day return greater than zero; and
   - at least 50% of the eligible universe above its own 20-day moving average.

The rule and all thresholds were committed before the full historical run. Validation
and shadow results cannot alter them.

## Required ablation

The report compares five policies:

- frozen factor Top10 on every day;
- frozen factor Top5 on every day;
- frozen factor Top5 on the fixed risk-on days;
- rolling ridge-rank Top5 on every day;
- rolling ridge-rank Top5 on the fixed risk-on days (primary).

The same-risk-day factor Top5 comparison isolates stock selection from the market gate.
The all-day comparisons show the effect of concentration and abstention.

## Anti-mechanical-improvement gates

A smaller book is not automatically an improvement. The primary must independently pass
in both validation and shadow:

- at least 25 signal days;
- at least eight non-overlapping ten-day events;
- basket win-rate lift of at least five percentage points over factor Top10;
- positive mean ten-day return;
- positive costed cumulative return at the preregistered 30 bps round trip;
- positive win-rate lift over factor Top5 on exactly the same risk-on days; and
- HAC t-statistic of at least 1.65 for the same-day return difference.

HAC uses lag ten because adjacent ten-day labels overlap. Monthly results and independent
events remain visible even when the aggregate headline looks good.

## Interpretation limit

The validation and shadow windows have been viewed in prior studies. This experiment is
therefore a pseudo-OOS diagnostic of a frozen algorithm, not a pristine final holdout.
Current-master survivorship, incomplete historical ST coverage and raw unadjusted prices
also remain. Even a pass could only justify a new forward shadow hypothesis.

## Running

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_perception_xalpha_winrate_v2.py
```

Outputs are written under:

```text
outputs/edge_research/perception_xalpha_winrate_v2/<run_id>/
```

The script does not read or write a trading config, strategy overlay, broker state,
position, order, risk gate, execution lock or `build_decision()`.

## Preregistered historical run

Run `run_20260803_preregistered_v2` used data from 2019-10-09 through 2026-07-31.
The primary policy produced:

| Period | Signal days | Independent events | 10d mean | Win rate | Costed cumulative |
|---|---:|---:|---:|---:|---:|
| Train walk-forward | 180 | 30 | +1.81% | 65.56% | +25.29% |
| Validation | 38 | 5 | +1.63% | 63.16% | +4.55% |
| Shadow | 16 | 3 | -0.20% | 56.25% | -0.58% |

The rolling rank model's mean daily rank IC remained positive in train, validation and
shadow (`0.0419 / 0.0491 / 0.0295`), but its HAC evidence weakened (`2.90 / 1.46 /
0.96`). On the same risk-on dates it beat factor Top5 in shadow, but underperformed it in
validation. Shadow activity also failed both the 25-signal-day and eight-independent-event
minimums.

The result is therefore `reject_for_trading_keep_diagnostics`. It demonstrates why win
rate alone is unsafe: 56.25% winning shadow baskets still had negative mean return because
the losing baskets were larger. No threshold, model, trading configuration or gate was
changed after observing this result.
