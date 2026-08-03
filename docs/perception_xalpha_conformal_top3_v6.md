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
