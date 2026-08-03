# Perception-XAlpha V6: Top10 return intervals and conservative Top3

## Status

Research-only / shadow-only / not a trading signal. V6 cannot write trading
configuration, strategy overlays, orders, positions, risk gates, or
`build_decision()`.

## Frozen hypothesis

V4 showed that one absolute-return model was poorly calibrated across market
regimes. V5 showed that a recent nested reliability audit did not make that
head stable in the reused shadow period. V6 therefore makes one narrower test:

1. estimate the common ten-session A-share market return from four past-only
   market variables;
2. estimate each stock's residual ten-session alpha from the frozen V2
   cross-sectional inputs;
3. add the two estimates;
4. derive an asymmetric empirical return interval from a strictly earlier
   calibration segment; and
5. show intervals for the frozen V2 Top10, but select no more than three only
   when the lower bound is strictly above the frozen 30 bps round-trip cost.

Zero, one, or two selections are valid. The system must never fill the Top3
quota merely because three names are requested. “Lower bound above cost” is an
empirical confidence rule, not a guarantee of profit.

## Time isolation

Every rolling fold is ordered as follows:

`base fit -> 10-day purge -> 63-day calibration -> 10-day purge -> 63-day reliability audit -> 10-day outer purge -> prediction block`

The two Ridge models and their 1%/99% target clipping are fit only on the base
segment. Signed residual interval offsets are fit only on calibration data.
The reliability audit may enable or disable the interval for later prediction
dates, but may never fit a model, a bound, or a threshold. Validation and
shadow labels may not tune anything.

## Interval interpretation

The interval is a date-equal-weighted empirical split-residual interval. Each
calibration date receives total weight one, preventing a date with more
securities from dominating the residual distribution. Because stock returns
are serially and cross-sectionally dependent, V6 does not claim exact
finite-sample conformal coverage. It reports coverage and lower-bound
violations on a later never-fit audit segment and fails closed when reliability
requirements are not met.

Relevant methodology:

- Chernozhukov, Wüthrich, and Zhu (2018), exact and robust conformal inference
  under dependent data.
- Zaffran et al. (2022), adaptive conformal prediction for time series.
- Xu and Xie (2023), sequential predictive conformal inference for time
  series.

## Frozen evaluation

The primary policy is `conformal_top3_positive_lower_bound`. Validation and
shadow must both have enough signal days and independent events, positive
costed return, at least 60% basket win rate, acceptable interval coverage, no
worse downside-tail rate than V2 Top3, and a positive same-day return lift with
HAC t-statistic at least 1.65. Each selected slot also needs at least ten
observations, positive mean return, and win rate above 50%.

Historical success still cannot promote V6 because the evaluation windows were
already inspected in earlier versions. A fresh forward shadow cohort would be
required before any separate paper-trading proposal.

## Frozen historical result (2026-08-03)

Run: `run_20260803_preregistered_conformal_top3_v6`  
Data: 2019-10-09 through 2026-08-03  
Prediction rows: 57,050  
Result: **rejected for trading; zero eligible Top3 names on the latest date.**

None of the 55 rolling folds passed the complete reliability gate. Interval
coverage passed in 36 folds, the width cap passed in 43, and rank-IC HAC passed
in only 12. Most importantly, no fold had the preregistered minimum of ten
audit observations whose lower return bound exceeded cost. The latest fold's
audit coverage was 57.54%, lower-bound violation rate was 34.26%, rank-IC HAC
t was -1.0241, and there were zero positive-lower-bound observations.

| Period | Policy | 10d mean | Win rate | Tail-loss rate | Costed cumulative |
|---|---|---:|---:|---:|---:|
| Validation | V2 Top3 | 1.5596% | 57.14% | 12.70% | 15.75% |
| Validation | point/lower-bound-ranked Top3 | 2.0113% | 65.08% | 8.73% | 22.35% |
| Shadow | V2 Top3 | -0.1309% | 45.22% | 36.52% | -5.05% |
| Shadow | point/lower-bound-ranked Top3 | 0.1476% | 40.35% | 34.21% | -0.97% |
| Validation | positive-lower-bound Top3 | no selections | — | — | 0.00% |
| Shadow | positive-lower-bound Top3 | no selections | — | — | 0.00% |

The attractive validation result did not survive the reused shadow period:
win rate fell to 40.35% and the costed book remained negative. Top10 interval
coverage also fell from 82.98% in validation to 46.73% in shadow, while the
lower-bound violation rate rose from 7.16% to 43.46%. This is a regime/calibration
failure, not evidence that a stricter threshold would guarantee profit.

For 2026-08-03, point estimates for the Top10 ranged from 3.87% to 5.52%, but
their conservative lower bounds ranged from -3.83% to -2.18%; all ten therefore
failed the 0.30% cost hurdle. The interval width was 14.61 percentage points.
Because V6 applies one fold-level signed-residual offset to every stock, lower-
bound ordering is identical to point-forecast ordering. A future version would
need a separately preregistered, past-only conditional scale model to test
whether stock-specific uncertainty adds ranking information. That experiment
was not performed here.
