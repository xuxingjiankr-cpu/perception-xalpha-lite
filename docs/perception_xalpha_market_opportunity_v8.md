# Perception-XAlpha V8: calibrated market-opportunity gate

Status: **preregistered research-only / shadow-only / not a trading signal**.

## Question

Can the frozen V2 cross-sectional Ridge ranker keep choosing the same Top3 stocks,
while a separate probability model avoids ten-session market environments in which a
long-only A-share basket is unlikely to clear the fixed 30 bp round-trip cost?

V8 is not another stock-return model and does not change V2 ranking. Its only increment
is the replacement of V2's deterministic trend-and-breadth gate with one calibrated
market-opportunity probability. This separates relative stock selection from the
absolute long/cash decision.

## Frozen design before the historical run

- Horizon: 10 trading days; signal at day `t` close, hypothetical entry at `t+1` open.
- Candidate pool: frozen V2 Top50; selected names: frozen V2 ranks 1-3.
- Label: equal-weight eligible-A-share future 10-day return strictly above 0.30%.
- Past-only features: 20/60-day market return, 20-day market volatility and 20-day
  breadth. No stock future return enters the model.
- Model: L2 logistic regression, `C=0.1`, training-median imputation and training-only
  standardization.
- Calibration: Platt logistic regression on a separate 63-day calibration segment.
- Reliability audit: a later, never-fit 63-day segment.
- Outer rolling fit: 756 days, minimum 504, refit every 21 days, with a 10-day purge.
- Nested chronology: base fit -> 10-day purge -> 63-day calibration -> 10-day purge
  -> 63-day audit -> outer 10-day purge -> prediction block.
- Selection threshold: calibrated probability >= 60%, and only when every audit
  reliability check passes. Zero selections are valid; the policy never fills a quota.

The reliability audit must beat the frozen base-rate forecast on Brier and LogLoss,
have AUC >= 0.55, ECE <= 0.15, positive calibration slope, and at least 10 audit dates
in the >=60% bucket with realized hit rate >=60%.

## Evidence required

Validation and shadow must independently have at least 20 signal days, 8 independent
10-day events and 15% calendar coverage. The Top3 basket must have win rate >=60%,
positive mean return and positive costed cumulative return. Each rank slot must have at
least 10 observations, positive mean and win rate above 50%.

To prevent an abstention illusion, the same frozen V2 Top3 basket is compared between
gated and excluded dates. The gated cohort must have higher mean return with HAC
`t >= 1.65`, no lower win rate and no worse 3% tail-loss rate. Aggregate calibrated
probability metrics must also beat the frozen prior and satisfy AUC/ECE thresholds.

## Interpretation boundary

The validation and shadow windows have already been inspected by earlier V1-V7 work.
Even a pass is only a forward-shadow hypothesis; it cannot change production, place an
order, alter a risk gate or claim guaranteed profit. A failure ends this exact V8
specification. It must not be repaired by searching probability thresholds or adding
models on the same evaluation windows.

## Result

Historical run completed once as
`run_20260803_preregistered_market_opportunity_v8` over 2019-10-09 through
2026-08-03 (57,050 stock-date prediction rows; 55 rolling market-model folds).

Verdict: **rejected for trading; retain diagnostics only**.

Only 3/55 fold audits passed every reliability condition. Individual pass counts were:

| Audit condition | Passing folds |
|---|---:|
| minimum audit sample | 55/55 |
| Brier below frozen prior | 26/55 |
| LogLoss below frozen prior | 26/55 |
| AUC >= 0.55 | 24/55 |
| ECE <= 0.15 | 17/55 |
| positive calibration slope | 31/55 |
| >=10 high-probability dates | 27/55 |
| high-probability hit rate >=60% | 15/55 |

The aggregate calibrated probability degraded sharply outside the earlier training
walk-forward region:

| Period | Brier / prior | LogLoss / prior | AUC | ECE |
|---|---:|---:|---:|---:|
| train walk-forward | 0.2609 / 0.2590 | 0.7550 / 0.7112 | 0.5900 | 0.0970 |
| validation | 0.3156 / 0.2661 | 0.8341 / 0.7255 | 0.4166 | 0.3525 |
| shadow | 0.3292 / 0.2465 | 0.8747 / 0.6862 | 0.4136 | 0.3204 |

The reliability-gated primary policy selected zero validation and shadow dates. This
was not merely an over-strict gate hiding a useful model: bypassing reliability and
using probability >=60% gave the following ablation.

| Period | Policy | Days | 10d mean | Win | 3% tail | Costed cumulative |
|---|---|---:|---:|---:|---:|---:|
| validation | V2 Top3 all dates | 126 | +1.5596% | 57.14% | 12.70% | +15.75% |
| validation | probability threshold only | 45 | +1.9683% | 55.56% | 11.11% | +6.43% |
| shadow | V2 Top3 all labelled dates | 115 | -0.1309% | 45.22% | 36.52% | -5.05% |
| shadow | probability threshold only | 56 | -0.2327% | 44.64% | 32.14% | -4.74% |

The threshold reduced tail frequency, but did not improve win rate in either period
and made shadow mean return worse. Validation-only mean improvement therefore did not
transfer. On 2026-08-03 the calibrated market-opportunity probability was 39.61%, the
fold was reliability-disabled, and the research policy correctly selected zero names.

No parameter, threshold or production path was changed after observing these results.
The exact V8 branch should not be repaired on these reused windows. A next hypothesis
must use a different economic mechanism and be separately preregistered; the current
evidence does not support deploying a market-timing probability gate.
