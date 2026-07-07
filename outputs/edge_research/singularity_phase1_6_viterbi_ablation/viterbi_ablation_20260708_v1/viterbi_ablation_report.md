# Singularity Phase 1.6 Viterbi ablation report

- Run ID: `viterbi_ablation_20260708_v1`
- Source commit at run: `e66bd2f`
- Status: `diagnostic_only`
- Research-only: `True`
- Shadow-only: `True`

Online Viterbi here means prefix endpoint decoding only. It does not use full-sequence backtracking, forward-backward smoothing, or future observations.

## Viterbi feature audit

- rows: `24000`
- firstTimestamp: `2024-06-12 09:35:00`
- lastTimestamp: `2026-07-03 15:00:00`
- sessionReset: `True`
- prefixOnly: `True`
- fullSequenceBacktrackingUsed: `False`
- forwardBackwardSmoothingUsed: `False`
- prefixInvarianceCheckPassed: `True`
- endpointStateSwitchRate: `0.45951914663110965`
- meanEndpointConfidence: `0.8538066283163921`
- meanStateAgeBars: `2.2940416666666668`
- filteredAgreementRate: `0.9801352874859075`
- phase2AConfigValidated: `True`

## Calibrated purged walk-forward metrics

| h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | constant_prior | 17,598 | 0.042567 | 0.184607 | 0.518600 | 0.002942 | 0 |
| 5 | current_hmm | 17,598 | 0.042280 | 0.181716 | 0.566741 | 0.003532 | 0 |
| 5 | current_hmm_ews | 17,598 | 0.041341 | 0.171614 | 0.683698 | 0.002943 | 8 |
| 5 | ews_baseline | 17,598 | 0.041316 | 0.171730 | 0.682553 | 0.003668 | 9 |
| 5 | hmm_ews_viterbi | 17,598 | 0.041376 | 0.171925 | 0.679989 | 0.002853 | 8 |
| 5 | hmm_viterbi | 17,598 | 0.042318 | 0.182058 | 0.559285 | 0.003994 | 0 |
| 10 | constant_prior | 13,202 | 0.066046 | 0.256756 | 0.543243 | 0.004119 | 0 |
| 10 | current_hmm | 13,202 | 0.065212 | 0.250632 | 0.614857 | 0.006330 | 0 |
| 10 | current_hmm_ews | 13,202 | 0.063649 | 0.238837 | 0.708251 | 0.006031 | 19 |
| 10 | ews_baseline | 13,202 | 0.063668 | 0.238984 | 0.707972 | 0.006390 | 19 |
| 10 | hmm_ews_viterbi | 13,202 | 0.063923 | 0.240216 | 0.702949 | 0.006695 | 23 |
| 10 | hmm_viterbi | 13,202 | 0.065302 | 0.251164 | 0.609569 | 0.006348 | 0 |

## Candidate gates

- 5-bar: metric wins 1/4; folds 1/7; months 3/13; ETF categories 3; high-risk n=8; Brier delta 0.00003491 CI [-0.00000946, 0.00007936]; pass=`False`.
- 10-bar: metric wins 0/4; folds 1/7; months 4/13; ETF categories 0; high-risk n=23; Brier delta 0.00027321 CI [0.00013258, 0.00042278]; pass=`False`.

## Conclusion

- Viterbi helps HMM+EWS under fixed gate: `False`
- Production or forward-task use allowed: `False`
- Recommended action: `retain_as_research_shadow_only_if_increment_is_stable_else_drop`

This result is not authorization to change paper trading. Any trading use requires a separate preregistration, fresh forward sample, costed replay on the actual decision population, and explicit economic-materiality gates.