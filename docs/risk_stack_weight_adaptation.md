# Risk Stack Phase 1.8 weight adaptation

## Objective

Phase 1.7 showed that the equal-weight BOCPD/DMD/HMM/Hawkes stack did not
validate as a stable risk signal.  Phase 1.8 asks a narrower question:

> If all model features remain fixed, can train-only walk-forward weighting
> improve the stack versus equal weights?

This is a research-only diagnostic.  It does not create a BUY/SELL rule, does
not affect paper trading, and does not write any production probability
artifact.

## Boundary

Allowed:

- use Phase 1.7 causal feature outputs;
- normalize each component with training-fold quantiles only;
- choose from a pre-registered non-negative weight grid;
- calibrate scores with training-fold bucket hit rates;
- evaluate only on purged future folds.

Forbidden:

- continuous full-sample weight fitting;
- negative weights or inverse signals;
- use of Phase 1.5 forward ledgers;
- use of paper-trading orders or PnL;
- changes to live config, overlays, `build_decision()`, BUY/SELL gates,
  position sizing, risk gates, or triple execution locks.

## Walk-forward design

- Test period: `2025-07-01` to `2026-07-03`.
- Fold frequency: calendar month.
- Training data for each fold: all rows strictly before the test month, with a
  purge of `10` 5-minute bars.
- Primary labels: `turning_point_5`, `turning_point_10`.
- Objective: average training AUC across the primary horizons.
- Candidate set: fixed in `configs/research/risk_stack_weight_adaptation_v1.json`.

## Interpretation

If the adaptive selector repeatedly chooses `hmm_only`, the correct conclusion
is not “the whole stack works.”  It means BOCPD, DMD and Hawkes should be
down-weighted in the current research sample.  That can improve diagnostics
relative to equal weights, but it does not validate a multi-model architecture.

Any trading use would still require a separate costed replay and fresh forward
validation.
