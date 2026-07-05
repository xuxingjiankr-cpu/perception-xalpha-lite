# Singularity Phase 2A historical offline plan

## Scope

Phase 2A is a historical feasibility study. It is not a Phase 1.5 model
revision, a production upgrade, an online inference service, or a trading
strategy.

The study compares five fixed variants on identical rows:

1. Phase 1 causal baseline;
2. baseline + generic EWS;
3. EWS + simplified classic LPPLS diagnostics;
4. EWS + fixed-window linear delay-DMD residual diagnostics; and
5. EWS + LPPLS + DMD.

Deep LPPLS, kernel/deep Koopman, Hawkes, SOC avalanche features, surrogate ML,
dashboards and all trading integration are excluded.

## Data audit first

The primary source is the local mootdx 5-minute OHLCV archive for 18 ETFs.
The audit runs before model construction and reports:

- date and symbol coverage;
- per-symbol missing and duplicate bars;
- invalid OHLC relationships and abnormal jumps;
- 1-minute, daily, tick, L2 and IOPV availability;
- timezone and adjustment metadata limitations;
- effective label samples and independent turning episodes;
- fixed-rule LPPLS windows; and
- equal-interval DMD windows.

If a preregistered audit gate fails, the run produces audit artifacts and
skips model fitting.

## Fixed simplified LPPLS

LPPLS is treated only as a local log-price shape diagnostic. For each fixed
decision timestamp, the script fits the classic linearized LPPLS basis on
past-only nested 16/20/24-bar windows. The nonlinear values are not optimized
against labels: `m`, angular frequency and critical-time offsets come from the
small grid frozen in the config.

The output includes normalized residual, trend eligibility, fitted coefficient
scale, and the distribution/stability of `tc`, `m` and frequency across nested
windows. No single critical time is presented as a forecast.

Because 24 five-minute bars represent only two trading hours, this can test
whether LPPLS geometry carries incremental classification information; it
cannot validate the economic interpretation of a long-horizon bubble model.

## Fixed linear delay-DMD

The Koopman feasibility leg is a linear delay-DMD residual model, not a kernel
or neural Koopman model. A 24-bar past-only window with delay dimension three
uses own return, current market return, current relative strength and past
volatility.

Variable means and standard deviations are fitted only through 2025-06-30 and
then frozen. The diagnostics are one-step held-out reconstruction residual,
training reconstruction error and spectral radius.

## Labels and splitting

Turning labels reuse the Phase 1 definition and remain in a separate offline
table. Horizons 5 and 10 are model targets; 20 and 30 are audit-only; 60 stays
null.

Walk-forward evaluation starts 2025-07-01. Each test block contains 40 complete
trading dates. Before both calibration and test boundaries, one complete
trading day is embargoed and 30 rows per symbol are purged. The final 20
eligible training dates form the Platt-calibration set. All variants forecast
exactly the same test population.

## Bias controls

- Features are strictly past-only.
- Future prices appear only in the offline label table.
- LPPLS scanning rules and DMD dimensions are fixed before evaluation.
- EWS/DMD normalization and volatility-regime thresholds are fitted only in
  the initial training period.
- Results are reported by fold, month, ETF category and volatility regime.
- Five variants create multiple-selection risk; the report does not hide it.
- Equal row counts are required so an apparent gain cannot come from removing
  difficult observations.
- Phase 1.5 forward files and frozen model are prohibited inputs and writes.

## Phase 2B discussion gate

Phase 2B may be discussed only when at least one LPPLS/DMD candidate improves
over EWS at both 5 and 10 bars, improves at least three of Brier, LogLoss, AUC
and ECE, persists across a majority of folds and months and at least two ETF
categories, has a sufficiently populated and non-overconfident high-risk
bucket, passes causal tests, and uses the same sample population.

Passing this historical gate would permit a separate preregistration
discussion only. It would not alter Phase 1.5 or authorize online or trading
integration.
