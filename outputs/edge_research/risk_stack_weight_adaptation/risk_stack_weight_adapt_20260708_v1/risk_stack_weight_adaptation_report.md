# Risk Stack Phase 1.8 weight adaptation

- Run ID: `risk_stack_weight_adapt_20260708_v1`
- Source commit at run: `decac9b`
- Status: `diagnostic_only`
- Research-only: `True`
- Paper integration allowed: `False`

This run tests train-only walk-forward weights for the Phase 1.7 BOCPD/DMD/HMM/Hawkes risk-context stack. It does not create a trading rule.

## Data audit

- rows: `44350`
- symbols: `18`
- tradingDays: `500`
- walkForwardStart: `2025-07-01`
- walkForwardEnd: `2026-07-03`
- initialTrainingEnd: `2025-06-30`
- folds: `26`
- dmdCompleteRows: `26610`
- phase17PaperIntegrationAllowed: `False`

## Walk-forward selection

| population | folds | dominant | candidate counts |
|---|---:|---|---|
| dmd_complete | 13 | dmd_only | `{'dmd_only': 13}` |
| full_session | 13 | hmm_only | `{'hmm_only': 13}` |

## Aggregate OOS metrics

| population | mode | label | n | AUC | Brier | LogLoss | ECE | top hit | lift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| dmd_complete | adaptive | turning_point_5 | 8,816 | 0.5207 | 0.03453 | 0.15518 | 0.0015 | 0.0385 | 1.079 |
| dmd_complete | adaptive | turning_point_10 | 4,408 | 0.4471 | 0.05378 | 0.21907 | 0.0045 | 0.0340 | 0.597 |
| dmd_complete | dmd_only | turning_point_5 | 8,816 | 0.5207 | 0.03453 | 0.15518 | 0.0015 | 0.0385 | 1.079 |
| dmd_complete | dmd_only | turning_point_10 | 4,408 | 0.4471 | 0.05378 | 0.21907 | 0.0045 | 0.0340 | 0.597 |
| dmd_complete | equal | turning_point_5 | 8,816 | 0.4727 | 0.03453 | 0.15517 | 0.0016 | 0.0261 | 0.730 |
| dmd_complete | equal | turning_point_10 | 4,408 | 0.4481 | 0.05384 | 0.21950 | 0.0052 | 0.0339 | 0.596 |
| dmd_complete | hmm_only | turning_point_5 | 8,816 | 0.4623 | 0.03456 | 0.15559 | 0.0022 | 0.0283 | 0.793 |
| dmd_complete | hmm_only | turning_point_10 | 4,408 | 0.4898 | 0.05389 | 0.21991 | 0.0063 | 0.0711 | 1.249 |
| full_session | adaptive | turning_point_5 | 17,632 | 0.5241 | 0.04235 | 0.18207 | 0.0050 | 0.0465 | 1.051 |
| full_session | adaptive | turning_point_10 | 13,224 | 0.5274 | 0.06602 | 0.25679 | 0.0077 | 0.0938 | 1.322 |
| full_session | equal | turning_point_5 | 17,632 | 0.4875 | 0.04229 | 0.18140 | 0.0047 | 0.0282 | 0.639 |
| full_session | equal | turning_point_10 | 13,224 | 0.4974 | 0.06602 | 0.25672 | 0.0063 | 0.0428 | 0.603 |
| full_session | hmm_only | turning_point_5 | 17,632 | 0.5241 | 0.04235 | 0.18207 | 0.0050 | 0.0465 | 1.051 |
| full_session | hmm_only | turning_point_10 | 13,224 | 0.5274 | 0.06602 | 0.25679 | 0.0077 | 0.0938 | 1.322 |

## Conclusion

- Adaptive full-session weights improve equal AUC on both horizons: `True`
- Adaptive full-session weights improve equal Brier on both horizons: `False`
- Dominant full-session candidate: `hmm_only`
- Multi-model blend validated: `False`
- Costed replay justified now: `False`
- Recommended next step: `keep_hmm_as_primary_shadow_risk_context_and_collect_forward_data`

If adaptation collapses to HMM-only, it means the other components should be down-weighted in research diagnostics. It is not approval to connect the score to trading.