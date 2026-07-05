# Singularity Phase 1.6 HMM physics-feature report

- Run ID: `model_20260705_break_20260612_v2`
- Source commit at run: `3d5dcf3`
- Status: `diagnostic_only`
- User break date: `2026-06-12`
- Post-break trading days: 15
- Post-jump HMM evaluated: `False`

Research-only. LPPLS/DMD are auxiliary covariates, never independent signals or BUY/SELL gates.

## Observed

- Full feature audit rows: 44,350.
- Full-session same-sample coverage: 99.77%.
- DMD-complete same-sample coverage: 39.89%.
- The complete DMD auxiliary vector retains only 14:15 and 14:40 decisions; 13:50 has no prior same-session spectral-radius drift.
- Post-break period contains 15 trading days.
- The existing 3-state GaussianHMM1D was reused without state search.

## Calibrated purged walk-forward metrics

| population | h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dmdCompleteLateSession | 5 | constant_prior | 4,396 | 0.028907 | 0.137516 | 0.465413 | 0.001590 | 0 |
| dmdCompleteLateSession | 5 | current_hmm | 4,396 | 0.028939 | 0.137736 | 0.485874 | 0.001664 | 0 |
| dmdCompleteLateSession | 5 | current_hmm_ews | 4,396 | 0.028812 | 0.134882 | 0.580349 | 0.002639 | 3 |
| dmdCompleteLateSession | 5 | ews_baseline | 4,396 | 0.028738 | 0.133977 | 0.591525 | 0.001639 | 2 |
| dmdCompleteLateSession | 5 | hmm_dmd | 4,396 | 0.028922 | 0.137309 | 0.503491 | 0.001988 | 0 |
| dmdCompleteLateSession | 5 | hmm_ews_dmd | 4,396 | 0.028812 | 0.134961 | 0.579608 | 0.002421 | 1 |
| dmdCompleteLateSession | 5 | hmm_ews_lppls_dmd | 4,396 | 0.028821 | 0.133671 | 0.612671 | 0.003621 | 7 |
| fullSession | 5 | constant_prior | 17,598 | 0.042567 | 0.184607 | 0.518600 | 0.002942 | 0 |
| fullSession | 5 | current_hmm | 17,598 | 0.042280 | 0.181716 | 0.566741 | 0.003532 | 0 |
| fullSession | 5 | current_hmm_ews | 17,598 | 0.041341 | 0.171614 | 0.683698 | 0.002943 | 8 |
| fullSession | 5 | ews_baseline | 17,598 | 0.041316 | 0.171730 | 0.682553 | 0.003668 | 9 |
| fullSession | 5 | hmm_ews_lppls | 17,598 | 0.041238 | 0.170786 | 0.693221 | 0.002737 | 12 |
| fullSession | 5 | hmm_lppls | 17,598 | 0.042222 | 0.180920 | 0.579969 | 0.003990 | 0 |
| fullSession | 10 | constant_prior | 13,202 | 0.066046 | 0.256756 | 0.543243 | 0.004119 | 0 |
| fullSession | 10 | current_hmm | 13,202 | 0.065212 | 0.250632 | 0.614857 | 0.006330 | 0 |
| fullSession | 10 | current_hmm_ews | 13,202 | 0.063649 | 0.238837 | 0.708251 | 0.006031 | 19 |
| fullSession | 10 | ews_baseline | 13,202 | 0.063668 | 0.238984 | 0.707972 | 0.006390 | 19 |
| fullSession | 10 | hmm_ews_lppls | 13,202 | 0.063725 | 0.239077 | 0.707438 | 0.005821 | 29 |
| fullSession | 10 | hmm_lppls | 13,202 | 0.065107 | 0.250037 | 0.619532 | 0.005906 | 2 |

## Fixed success gates

### lppls

- 5-bar: metric wins 4/4; folds 4/7; months 9/13; high-risk n=12; Brier delta -0.00010479 CI [-0.00027797, 0.00007305]; pass=`False`.
- 10-bar: metric wins 1/4; folds 2/7; months 7/13; high-risk n=29; Brier delta 0.00007770 CI [-0.00020756, 0.00034399]; pass=`False`.
- Helps HMM under the fixed gate: `False`

### dmd

- 5-bar: metric wins 1/4; folds 4/7; months 7/13; high-risk n=1; Brier delta -0.00000134 CI [-0.00006495, 0.00005705]; pass=`False`.
- 10-bar: `not_evaluable_insufficient_same_session_coverage`; pass=`False`.
- Helps HMM under the fixed gate: `False`

### combined

- 5-bar: metric wins 2/4; folds 4/7; months 8/13; high-risk n=7; Brier delta 0.00000457 CI [-0.00071552, 0.00070924]; pass=`False`.
- 10-bar: `not_evaluable_insufficient_same_session_coverage`; pass=`False`.
- Helps HMM under the fixed gate: `False`

## Required conclusions

- HMM improved by LPPLS: `False`
- LPPLS forward-retest status: `retain_frozen_hypothesis_promising_5bar_not_proven`
- HMM improved by Koopman/DMD: `False`
- Combined physics features help HMM: `False`
- Improvement survives same-sample comparison: `False`
- Post-jump HMM improved: `not_evaluated_insufficient_post_break_days`
- Time-decay HMM improved: `not_evaluated_half_life_not_preregistered`
- HMM state matrix/durations/entropy changed: `false` by design.
- Remains research-only: `true`

DMD conclusions apply only to the explicitly reported DMD-complete late-session population and cannot be extrapolated to the full session.
