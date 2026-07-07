# Singularity Phase 1.6: causal Viterbi ablation

## Scope

This research tests whether causal online Viterbi decoding adds useful state
stability information beyond the existing HMM filtered probability features.

It does not change the HMM, retrain HMM states, fit a post-jump HMM, alter
Phase 1.5, or connect anything to paper trading.

## Causal definition

For each trading day, the decoder resets at the session open and processes the
market return sequence as a prefix:

```text
observations[0:t] -> Viterbi endpoint state at t
```

The implementation records only the endpoint state and prefix-derived path
diagnostics available at time `t`. It does not use full-sequence backtracking,
forward-backward smoothing, or future observations.

The added shadow features are:

- `viterbi_bear_state`;
- `viterbi_middle_state`;
- `viterbi_bull_state`;
- `viterbi_state_age_bars`;
- `viterbi_agrees_with_filtered_state`;
- `viterbi_path_transition_risk`;
- `viterbi_endpoint_confidence`;
- `viterbi_log_margin`.

## Test design

The study reuses the Phase 1.6 historical feature table and the frozen
GaussianHMM1D parameters recorded in the Phase 1.6 model result.

The variants are:

- `constant_prior`;
- `ews_baseline`;
- `current_hmm`;
- `current_hmm_ews`;
- `hmm_viterbi`;
- `hmm_ews_viterbi`.

The primary comparison is:

```text
hmm_ews_viterbi vs current_hmm_ews
```

on the same 5-bar and 10-bar turning labels, the same walk-forward dates, the
same purge/embargo settings, and the same calibrated logistic model family.

## Promotion boundary

This is research-only. Even if historical metrics improve, the result can only
justify a frozen forward-shadow hypothesis. It cannot:

- create BUY or SELL signals;
- amplify decision scores;
- change position sizing;
- change risk gates;
- change the SELL path;
- write `decision_probability_v1.json`;
- bypass the three execution locks.

Any trading use would require a separate preregistration with fresh forward
data, costed replay on the actual decision population, and explicit economic
materiality.
