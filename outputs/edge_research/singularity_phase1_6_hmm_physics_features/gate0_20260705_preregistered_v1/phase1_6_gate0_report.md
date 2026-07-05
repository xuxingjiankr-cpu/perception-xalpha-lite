# Singularity Phase 1.6 Gate 0 report

- Run ID: `gate0_20260705_preregistered_v1`
- Source commit at run: `3072121`
- Status: `exploratory_gate0_only`
- Gate 0 passed: `False`
- Formal HMM modeling allowed: `False`

## Observed

- Historical range is 2024-06-12 through 2026-07-03.
- The panel contains 500 trading days, 18 symbols and 425,760 bars.
- Independent 5/10-bar turning events are 6,180/5,607.
- LPPLS/DMD feature coverage is 100.00%/60.00%.
- Mootdx OHLCV has no bid/ask spread, source timestamp, explicit suspension flag or adjustment factor.

## Estimated

- Exploratory monthly change-point candidates are ranked by distribution shift only.
- Auxiliary preprocessing parameters are estimated only through 2025-06-30 and are not applied to a model.
- Feature correlations are descriptive and do not establish incremental information.

## Assumed

- Naive TDX timestamps are Asia/Shanghai.
- A full session has 48 ordered five-minute bars.
- The current 18-product universe is not a point-in-time historical constituent universe.

## Data sufficiency

- `full_history_hmm`: `data_volume_sufficient_but_not_run` — 500 trading days and 6,180/5,607 independent 5/10-bar events.
- `post_jump_hmm`: `blocked` — No independently preregistered break date.
- `time_decay_hmm`: `blocked` — Half-life and state specification are not preregistered.
- `multivariate_hmm`: `data_volume_sufficient_but_not_run` — Base OHLCV panel is broad enough, but its observation contract is not frozen.
- `lppls_hmm_auxiliary`: `coverage_feasible_but_not_run` — Observed fixed-window coverage is 100.00%.
- `dmd_hmm_auxiliary`: `coverage_insufficient` — Observed coverage is 60.00%; invalid rows are concentrated in earlier intraday slots.

## Passed tests

- Core historical data-quality and independent-event gates pass.
- LPPLS fixed-window auxiliary coverage passes the 90% threshold.
- Invalid auxiliary rows are retained with explicit validity/missingness fields.
- No profitability or OOS model metric is used to select a break candidate.
- Phase 1.5 forward ledger is neither read nor written.

## Failed tests

- Break date is not preregistered.
- DMD auxiliary coverage is below the fixed 90% Gate 0 threshold.

## Skipped items

- `deep_lppls`
- `kernel_koopman`
- `deep_koopman`
- `hawkes`
- `soc`
- `surrogate_ml`
- `reinforcement_learning`
- `online_inference`
- `dashboard`
- `live_trading_integration`
- `automatic_break_selection`
- `automatic_hmm_tuning`
- `all_hmm_model_fits`
- `post_jump_vs_full_history_hmm`
- `time_decay_hmm`
- `state_transition_matrix_comparison`
- `state_duration_comparison`
- `state_entropy_comparison`
- `turning_probability_calibration_comparison`

## Required conclusions

- HMM improved: `not_evaluated`
- Post-jump HMM improved: `not_evaluated_break_date_not_preregistered`
- Time-decay HMM improved: `not_evaluated_half_life_not_preregistered`
- LPPLS helped HMM: `not_evaluated`
- Koopman helped HMM: `not_evaluated`
- Improvement survives same-sample comparison: `not_evaluated`
- Remains research-only: `true`

The corrected Phase 2A interpretation is retained: LPPLS/DMD were not validated as independent trading signals, but their possible HMM auxiliary role remains an untested hypothesis.
