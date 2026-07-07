# Risk Stack Phase 1.7 audit

- Run ID: `risk_stack_p17_20260708_v2`
- Source commit at run: `4ba2bfa`
- Status: `diagnostic_only`
- Research-only: `True`
- Paper integration allowed: `False`

This audit tests BOCPD, DMD, HMM and Hawkes as risk context for exits and position de-risking. It is not a BUY/SELL model.

## Data audit

- phase16FeatureRows: `44350`
- symbols: `18`
- tradingDays: `500`
- walkForwardStart: `2025-07-01`
- walkForwardEnd: `2026-07-03`
- trainingStatisticsEnd: `2025-06-30`
- dmdCompleteRows: `26610`
- dmdCompleteCoverage: `0.6`
- hmmSourceFitEnd: `2025-06-30`

## Feature audits

- BOCPD prefix-invariant: `True`
- Hawkes events: `10599` / `44350` rows
- Hawkes event density: `0.2390`

## OOS bucket diagnostics

| label | score | n | AUC | base hit | top hit | lift | top n | months + | cats + |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| turning_point_5 | bocpd_only | 17,632 | 0.4843 | 0.0442 | 0.0348 | 0.787 | 1,780 | 5/13 | 1/5 |
| turning_point_5 | dmd_only | 8,816 | 0.5202 | 0.0357 | 0.0374 | 1.047 | 882 | 6/13 | 2/5 |
| turning_point_5 | hmm_transition_only | 17,632 | 0.5276 | 0.0442 | 0.0465 | 1.051 | 1,764 | 9/12 | 3/5 |
| turning_point_5 | hawkes_only | 17,632 | 0.4782 | 0.0442 | 0.0331 | 0.748 | 2,569 | 4/13 | 1/5 |
| turning_point_5 | risk_stack_full | 17,632 | 0.4856 | 0.0442 | 0.0265 | 0.598 | 1,776 | 2/13 | 0/5 |
| turning_point_5 | risk_stack_dmd_complete | 8,816 | 0.4708 | 0.0357 | 0.0260 | 0.729 | 883 | 4/13 | 2/5 |
| turning_point_10 | bocpd_only | 13,224 | 0.4853 | 0.0710 | 0.0548 | 0.772 | 1,331 | 3/13 | 0/5 |
| turning_point_10 | dmd_only | 4,408 | 0.4468 | 0.0569 | 0.0340 | 0.597 | 441 | 2/13 | 0/5 |
| turning_point_10 | hmm_transition_only | 13,224 | 0.5318 | 0.0710 | 0.0938 | 1.322 | 1,332 | 9/12 | 4/5 |
| turning_point_10 | hawkes_only | 13,224 | 0.4752 | 0.0710 | 0.0371 | 0.522 | 1,403 | 2/13 | 0/5 |
| turning_point_10 | risk_stack_full | 13,224 | 0.4949 | 0.0710 | 0.0428 | 0.603 | 1,332 | 1/12 | 0/5 |
| turning_point_10 | risk_stack_dmd_complete | 4,408 | 0.4469 | 0.0569 | 0.0295 | 0.518 | 441 | 1/12 | 2/5 |

## Conclusion

- BOCPD feasible: `True`
- Hawkes event density sufficient for fitting: `True`
- Full risk stack shows stable edge: `False`
- DMD-complete stack shows stable edge: `False`
- Costed replay justified now: `False`
- Recommended next step: `retain_audit_only_collect_more_forward_events`

The result cannot be used in paper trading without a separate costed replay and fresh forward validation.