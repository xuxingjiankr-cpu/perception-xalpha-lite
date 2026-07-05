# Singularity Phase 2A historical report

- Run ID: `phase2a_full_20260705_preregistered_v2`
- Source commit at run: `2071fa9`
- Status: `diagnostic_only`
- Data: 2024-06-12 through 2026-07-03
- Universe / days / rows: 18 / 500 / 425,760
- Independent turning events 5/10: 6,180 / 5,607
- LPPLS / DMD feasible windows: 44,350 / 44,350

Research-only and historical-only. Phase 1.5 is neither read nor written.

## Data and model feasibility

- Feature rows on the common population: 26,540
- LPPLS fixed-grid fits: 44,249 / 44,350; trend-eligible common rows: 574 (2.16%).
- DMD residual fits: 26,540 / 44,350.
- The common model population retains 59.84% of fixed decision windows. Requiring a complete 24-bar DMD state removes mainly earlier intraday windows; every ablation uses the same retained rows.
- Raw bars have no adjustment factor. One overnight split-like discontinuity and one intraday >10% jump are retained and disclosed.
- Tick data are absent; L2 and IOPV histories are too short and are not used.

### LPPLS stability and residual distributions

| diagnostic | n | mean | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| lppls_residual_mean | 26,540 | 0.480947 | 0.468900 | 0.266375 | 0.716251 |
| lppls_residual_std | 26,540 | 0.056483 | 0.047086 | 0.013870 | 0.112879 |
| lppls_tc_offset_median | 26,540 | 12.014921 | 18.000000 | 6.000000 | 18.000000 |
| lppls_tc_offset_std | 26,540 | 2.398088 | 0.000000 | 0.000000 | 5.656854 |
| lppls_m_median | 26,540 | 0.545365 | 0.700000 | 0.300000 | 0.700000 |
| lppls_m_std | 26,540 | 0.100419 | 0.188562 | 0.000000 | 0.188562 |
| lppls_omega_median | 26,540 | 7.567144 | 6.000000 | 6.000000 | 10.000000 |
| lppls_omega_std | 26,540 | 0.972223 | 1.885618 | 0.000000 | 1.885618 |
| lppls_b_scaled_mean | 26,540 | 0.019298 | 0.014095 | -2.393083 | 2.445373 |
| lppls_oscillation_amplitude | 26,540 | 0.238256 | 0.192842 | 0.089310 | 0.447755 |

No single `tc` is treated as a forecast; nested-window dispersion is retained as a feature.

### Linear DMD residual distributions

| diagnostic | n | mean | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| koopman_residual_norm | 26,540 | 0.732512 | 0.641099 | 0.267303 | 1.286457 |
| koopman_training_error | 26,540 | 0.296165 | 0.296582 | 0.219309 | 0.372065 |
| koopman_spectral_radius | 26,540 | 0.979015 | 0.978958 | 0.892730 | 1.051899 |
| koopman_effective_rank | 26,540 | 12.000000 | 12.000000 | 12.000000 | 12.000000 |
| koopman_instability | 26,540 | 0.018061 | 0.000000 | 0.000000 | 0.051899 |

## Purged walk-forward ablation

| h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | baseline | 8,795 | 0.035065 | 0.161815 | 0.488358 | 0.003418 | 0 |
| 5 | ews | 8,795 | 0.034849 | 0.155707 | 0.583042 | 0.004299 | 0 |
| 5 | ews_lppls | 8,795 | 0.034780 | 0.155146 | 0.593568 | 0.004682 | 4 |
| 5 | ews_koopman | 8,795 | 0.034871 | 0.155545 | 0.588198 | 0.004263 | 0 |
| 5 | ews_lppls_koopman | 8,795 | 0.034747 | 0.154596 | 0.603276 | 0.005036 | 2 |
| 10 | baseline | 4,399 | 0.054260 | 0.224986 | 0.480430 | 0.014396 | 0 |
| 10 | ews | 4,399 | 0.053758 | 0.219050 | 0.581035 | 0.013512 | 0 |
| 10 | ews_lppls | 4,399 | 0.053704 | 0.219212 | 0.584443 | 0.013728 | 0 |
| 10 | ews_koopman | 4,399 | 0.053681 | 0.218804 | 0.580445 | 0.012746 | 0 |
| 10 | ews_lppls_koopman | 4,399 | 0.053596 | 0.218489 | 0.588886 | 0.013028 | 0 |

## Probability buckets

| h | variant | bucket | n | mean p | hit rate |
|---:|---|---|---:|---:|---:|
| 5 | baseline | [0.0, 0.1) | 8,699 | 0.0384 | 0.0360 |
| 5 | baseline | [0.1, 0.2) | 95 | 0.1096 | 0.0211 |
| 5 | baseline | [0.2, 0.3) | 1 | 0.2238 | 0.0000 |
| 5 | ews | [0.0, 0.1) | 8,243 | 0.0339 | 0.0323 |
| 5 | ews | [0.1, 0.2) | 520 | 0.1272 | 0.0904 |
| 5 | ews | [0.2, 0.3) | 31 | 0.2338 | 0.0645 |
| 5 | ews | [0.3, 0.4) | 1 | 0.3372 | 0.0000 |
| 5 | ews_lppls | [0.0, 0.1) | 8,190 | 0.0328 | 0.0325 |
| 5 | ews_lppls | [0.1, 0.2) | 529 | 0.1282 | 0.0718 |
| 5 | ews_lppls | [0.2, 0.3) | 63 | 0.2339 | 0.1270 |
| 5 | ews_lppls | [0.3, 0.4) | 11 | 0.3337 | 0.1818 |
| 5 | ews_lppls | [0.4, 0.5) | 2 | 0.4428 | 0.5000 |
| 5 | ews_koopman | [0.0, 0.1) | 8,231 | 0.0335 | 0.0328 |
| 5 | ews_koopman | [0.1, 0.2) | 523 | 0.1282 | 0.0803 |
| 5 | ews_koopman | [0.2, 0.3) | 40 | 0.2294 | 0.0750 |
| 5 | ews_koopman | [0.3, 0.4) | 1 | 0.3187 | 0.0000 |
| 5 | ews_lppls_koopman | [0.0, 0.1) | 8,198 | 0.0327 | 0.0329 |
| 5 | ews_lppls_koopman | [0.1, 0.2) | 524 | 0.1299 | 0.0630 |
| 5 | ews_lppls_koopman | [0.2, 0.3) | 63 | 0.2363 | 0.1429 |
| 5 | ews_lppls_koopman | [0.3, 0.4) | 9 | 0.3250 | 0.2222 |
| 5 | ews_lppls_koopman | [0.4, 0.5) | 1 | 0.4683 | 1.0000 |
| 10 | baseline | [0.0, 0.1) | 4,382 | 0.0428 | 0.0571 |
| 10 | baseline | [0.1, 0.2) | 17 | 0.1166 | 0.0588 |
| 10 | ews | [0.0, 0.1) | 4,359 | 0.0433 | 0.0567 |
| 10 | ews | [0.1, 0.2) | 40 | 0.1245 | 0.1000 |
| 10 | ews_lppls | [0.0, 0.1) | 4,340 | 0.0429 | 0.0565 |
| 10 | ews_lppls | [0.1, 0.2) | 56 | 0.1214 | 0.1071 |
| 10 | ews_lppls | [0.2, 0.3) | 3 | 0.2335 | 0.0000 |
| 10 | ews_koopman | [0.0, 0.1) | 4,352 | 0.0434 | 0.0563 |
| 10 | ews_koopman | [0.1, 0.2) | 47 | 0.1276 | 0.1277 |
| 10 | ews_lppls_koopman | [0.0, 0.1) | 4,327 | 0.0429 | 0.0552 |
| 10 | ews_lppls_koopman | [0.1, 0.2) | 70 | 0.1218 | 0.1714 |
| 10 | ews_lppls_koopman | [0.2, 0.3) | 1 | 0.2071 | 0.0000 |
| 10 | ews_lppls_koopman | [0.3, 0.4) | 1 | 0.3204 | 0.0000 |

## Stability versus EWS

| candidate | h | metric wins (of 4) | improving folds | improving months | improving categories | high-risk n | pass |
|---|---:|---:|---:|---:|---:|---:|---|
| ews_lppls | 5 | 3 | 3/7 | 6/13 | 3 | 4 | `False` |
| ews_lppls | 10 | 2 | 2/7 | 7/13 | 3 | 0 | `False` |
| ews_koopman | 5 | 3 | 3/7 | 8/13 | 2 | 0 | `False` |
| ews_koopman | 10 | 3 | 3/7 | 8/13 | 3 | 0 | `False` |
| ews_lppls_koopman | 5 | 3 | 2/7 | 7/13 | 4 | 2 | `False` |
| ews_lppls_koopman | 10 | 4 | 4/7 | 10/13 | 4 | 0 | `False` |

The full group tables by month, ETF category and volatility regime are stored in `phase2a_result.json`. Improvements are not confined to one group, but none survives every fixed gate at both horizons.

## Phase 2B review

No candidate passes the fixed historical gate at both 5 and 10 bars; Phase 2B is not justified.

- Same forecast population: `True`
- Statistical preconditions: `False`
- Phase 2B discussion allowed: `False`
- Multiple-selection bias is present and explicitly retained.
- All variants forecast identical rows; no apparent gain is caused by reducing the forecast or trade population.
- The common population is nevertheless narrower than the full EWS population because DMD-state completeness filters early windows. The result establishes no full-session Phase 1 EWS increment.
- The additions show small aggregate gains over EWS, but fold instability and nearly empty high-risk buckets prevent a stable incremental-information claim.
- Features are past-only; future data appear only in the separate offline label table. Remaining risks are raw/unadjusted bars, inferred timezone, survivor-selected products and repeated use of historical data.

## Skipped items

- `deep_lppls`: outside fixed Phase 2A scope
- `kernel_koopman`: outside fixed Phase 2A scope
- `deep_koopman`: outside fixed Phase 2A scope
- `hawkes`: outside fixed Phase 2A scope
- `soc_avalanche`: outside fixed Phase 2A scope
- `surrogate_ml`: outside fixed Phase 2A scope
- `online_inference`: outside fixed Phase 2A scope
- `dashboard`: outside fixed Phase 2A scope
- `live_trading_integration`: outside fixed Phase 2A scope
- `horizon_60`: A 60-bar horizon exceeds one 48-bar A-share session and remains null.

20/30 bars are audit-only; 60 bars are skipped/null. Phase 2B is not justified. No online or trading integration is produced.
