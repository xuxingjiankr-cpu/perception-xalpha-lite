# Singularity Phase 1: preregistered shadow research

## Scope

This experiment is offline, research-only and shadow-only. It estimates
same-session ETF turning-point probabilities from:

1. the existing causal minute feature baseline;
2. the existing `GaussianHMM1D` implementation;
3. generic early-warning-statistic (EWS) features; and
4. a combined HMM + EWS ablation.

It does not place orders, modify positions, change a risk gate, alter the
three execution locks, write an overlay, change `build_decision()`, or write
`decision_probability_v1.json`.

Deep LPPLS, Koopman/DMD, Hawkes processes, SOC avalanche features, surrogate
ML, online inference, and any live integration are out of Phase 1.

## Frozen data and universe

- Source: `data/research/t0_opening/yahoo_60d_confirmed_t0_raw_5m.jsonl`
- Frequency: 5-minute bars
- Base training period: 2026-03-23 through 2026-05-20
- Walk-forward period: 2026-05-21 through 2026-06-18
- Universe: 50 confirmed-T0 ETFs selected only from base-training liquidity
  and coverage. Bond, money-market and cash-management asset classes are
  excluded before taking the top 50.

The walk-forward period has been reused by earlier research. Results therefore
remain diagnostic and cannot be described as a clean final OOS confirmation.

## Past-only features

All feature rows are computed at the completed-bar timestamp. No future price
is read while constructing a feature.

Baseline features:

- 1-, 3-, and 6-bar within-session returns;
- 1-bar return acceleration;
- 6-bar realized volatility;
- current cross-sectional market 1- and 3-bar return;
- current breadth;
- current relative-strength percentile;
- current cumulative-amount percentile; and
- current session fraction.

Generic EWS features use a frozen 12-bar trailing window:

- log trailing return variance;
- lag-1 trailing return autocorrelation;
- normalized slope of trailing squared returns; and
- current cross-sectional return dispersion.

EWS means and standard deviations are fitted through 2026-05-20 and frozen.
The four oriented component scores receive equal weight.

The HMM is imported from `scripts/research_hmm_nn_bl.py`; it is not
reimplemented. Its emission and transition parameters are fitted through
2026-05-20. Filtering is causal and resets at each session open. Regime
transition risk is the current filtered probability of changing state on the
next bar:

`1 - sum_s P(S_t=s | data through t) * P(S_(t+1)=s | S_t=s)`.

`singularity_score` is a transparent 50/50 average of EWS score and HMM
transition risk. It has no production meaning.

## Offline labels

Labels are written to a separate table and never joined back into historical
feature computation. For horizon `h`, a turning point is:

- reversal down: the current price is near its trailing 12-bar high and the
  next `h` same-session bars contain a sufficiently negative excursion; or
- reversal up: the current price is near its trailing 12-bar low and the next
  `h` same-session bars contain a sufficiently positive excursion.

The move threshold is frozen as the larger of 30 bps and 1.5 times the
past-only 6-bar volatility scaled by the square root of the horizon. Supporting
diagnostic labels record failed breakouts, trend exhaustion and stop-chasing.

Active horizons are 5, 10, 20 and 30 bars. At 5-minute frequency, 60 bars are
300 minutes and exceed the 240-minute A-share session. Horizon 60 is therefore
always null and explicitly reported as skipped; Phase 1 never crosses
overnight.

## Walk-forward protocol

The test period is divided into chronological four-trading-day blocks. For
each block and horizon:

1. only earlier trading dates can train the model;
2. the last 30 rows per ETF are purged before the test boundary;
3. the final six eligible training dates are a probability-calibration set;
4. the base-fit rows are again purged by 30 rows per ETF before calibration;
5. feature scaling and logistic parameters are fitted only on base-fit rows;
6. Platt calibration is fitted only on the later calibration dates; and
7. the entire test-date block remains disjoint.

Splitting by complete trading date prevents overlapping windows for one ETF
and one session from appearing on both sides. The 30-bar purge equals the
largest active label horizon.

## Outputs

Each run writes a new immutable directory:

`outputs/edge_research/singularity_phase1/<run_id>/`

The directory contains the frozen config snapshot, universe audit, separate
feature and label tables, walk-forward predictions, a research-only turning
probability JSONL artifact, machine-readable results, a Markdown report and a
run log. Existing replay and probability artifacts are never overwritten.

Run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_singularity_phase1.py
```

An explicit, previously unused run ID may be supplied with `--run-id`.

## Interpretation gate

The primary comparison is calibrated Brier score; LogLoss, AUC and ECE are
secondary. Probability-bucket hit rates and sample counts are always reported.
All four ablations score exactly the same test rows, so a metric difference
cannot be credited to mechanically suppressing trades. No trading decision is
filtered or executed in this experiment.

Phase 2 is not automatic. LPPLS or Koopman work is worth considering only if
HMM + EWS improves calibration consistently across horizons and folds without
materially worse tail buckets. Even then, genuinely new forward data is
required before any production discussion.
