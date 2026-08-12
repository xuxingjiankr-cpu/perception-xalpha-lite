# Twelve-Factor Rank Discrimination Audit

This research asks a narrow question: can a fixed nonlinear model or a genuine
cross-sectional learning-to-rank objective improve the daily Top10 produced by
the same twelve PIT factor ranks?

It does not search for new price-volume factors, tune factor weights, stretch
probabilities, or alter the forecast dashboard. All historical evaluation
windows have already been viewed, so even a positive result can only nominate a
separately preregistered fresh-forward candidate.

## Frozen comparison

- `linear`: the current guarded adaptive twelve-factor composite score used by
  the dashboard ranking.
- `nonlinear`: fixed shallow LightGBM return regressor for interactions.
- `lambdarank`: fixed LightGBM LambdaRank with same-day five-level relevance.
- `hybrid`: fixed 50/50 base-fit-standardized nonlinear and LambdaRank scores.

All four models use identical rows, the same twelve oriented factor ranks, the
same point-in-time neutral completion policy, and the same executable next-open
outcome. Model fitting, score calibration, audit, validation, and shadow are
strictly chronological with the existing purge gaps.

Each scalar score receives an independent calibration-period-only mapping to
expected return, up probability, and severe-loss probability. Probability
width is reported but never optimized or mechanically expanded.

## Acceptance

Relative to `linear`, a candidate must pass in both validation and shadow:

1. improve at least four of Top10 gross return, Top10 stock win rate, daily rank
   IC, up AUC, and up Brier;
2. improve Top10 gross return specifically; and
3. achieve a paired daily gross-return HAC t-statistic of at least 1.96.

The four declared trials are also charged to a CSCV/PBO audit, and each
candidate's shadow daily-return difference versus the current baseline receives
an approximate Deflated Sharpe adjustment. PBO must be at most 0.20 and DSR must
be significant at 10% before a historical pass can nominate a forward study.

The report also contrasts a model selected with the current period's outcomes
against one selected only from the preceding period. Their net-return gap is the
measured model-selection overfitting exposure.

No historical outcome can update the dashboard, trading configuration, order
path, position sizing, risk gate, or execution locks.
