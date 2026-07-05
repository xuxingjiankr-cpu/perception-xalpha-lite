# Singularity Phase 1 shadow report

- Status: **diagnostic_only**
- Run ID: `singularity_p1_20260705_preregistered_final`
- Generated: 2026-07-05T08:29:20+08:00
- Data: 2026-03-23 through 2026-06-18
- Bar interval: 5 minutes
- Universe: 50 training-selected, entry-eligible ETFs
- Feature rows: 77,649
- Label rows: 141,381

This output is research-only and shadow-only. It does not modify live trading, positions, orders, risk gates, execution locks, overlays, `build_decision()` or `decision_probability_v1.json`.

## Labels

| Horizon (bars) | Status | Samples | Positive rate |
|---:|---|---:|---:|
| 5 | active | 63,121 | 4.5833% |
| 10 | active | 38,886 | 6.9665% |
| 20 | active | 17,310 | 10.4391% |
| 30 | active | 7,521 | 10.1050% |
| 60 | skipped_null | 0 | null |

The primary target is a same-session reversal from a trailing 12-bar local extreme. The future window is used only to construct the separate offline label table. Horizon 60 is null because 300 minutes exceeds one A-share session.

## Walk-forward calibrated ablation

| Horizon | Variant | Brier | LogLoss | AUC | ECE | Forecasts |
|---:|---|---:|---:|---:|---:|---:|
| 5 | baseline | 0.050271 | 0.204163 | 0.617358 | 0.003308 | 19,269 |
| 5 | hmm | 0.050248 | 0.203971 | 0.619679 | 0.003671 | 19,269 |
| 5 | ews | 0.049816 | 0.200240 | 0.657091 | 0.002012 | 19,269 |
| 5 | hmm_ews | 0.049782 | 0.200051 | 0.658029 | 0.001704 | 19,269 |
| 10 | baseline | 0.067578 | 0.251668 | 0.699303 | 0.011378 | 10,724 |
| 10 | hmm | 0.067636 | 0.251941 | 0.697409 | 0.011609 | 10,724 |
| 10 | ews | 0.067015 | 0.249620 | 0.708003 | 0.013235 | 10,724 |
| 10 | hmm_ews | 0.067021 | 0.249706 | 0.706989 | 0.013386 | 10,724 |
| 20 | baseline | 0.092481 | 0.321465 | 0.705865 | 0.022152 | 5,666 |
| 20 | hmm | 0.092053 | 0.320312 | 0.706858 | 0.022627 | 5,666 |
| 20 | ews | 0.092359 | 0.320272 | 0.709586 | 0.024044 | 5,666 |
| 20 | hmm_ews | 0.091895 | 0.318938 | 0.710786 | 0.023867 | 5,666 |
| 30 | baseline | 0.107425 | 0.362225 | 0.645612 | 0.078021 | 2,097 |
| 30 | hmm | 0.105155 | 0.358687 | 0.641306 | 0.073984 | 2,097 |
| 30 | ews | 0.101248 | 0.351068 | 0.643440 | 0.062377 | 2,097 |
| 30 | hmm_ews | 0.100397 | 0.350158 | 0.641403 | 0.061331 | 2,097 |

## HMM + EWS probability buckets

| Horizon | Bucket | Count | Mean probability | Hit rate |
|---:|---|---:|---:|---:|
| 5 | [0.0, 0.1) | 17,497 | 4.6628% | 4.6579% |
| 5 | [0.1, 0.2) | 1,605 | 12.8881% | 11.4642% |
| 5 | [0.2, 0.3) | 141 | 23.2998% | 17.7305% |
| 5 | [0.3, 0.4) | 19 | 32.9894% | 26.3158% |
| 5 | [0.4, 0.5) | 7 | 42.6304% | 42.8571% |
| 10 | [0.0, 0.1) | 6,579 | 5.0124% | 3.7544% |
| 10 | [0.1, 0.2) | 3,771 | 13.6417% | 12.3575% |
| 10 | [0.2, 0.3) | 364 | 22.6006% | 25.5495% |
| 10 | [0.3, 0.4) | 8 | 33.5504% | 25.0000% |
| 10 | [0.4, 0.5) | 2 | 46.7033% | 0.0000% |
| 20 | [0.0, 0.1) | 3,162 | 4.4381% | 5.5977% |
| 20 | [0.1, 0.2) | 1,363 | 14.7876% | 13.2795% |
| 20 | [0.2, 0.3) | 757 | 24.6599% | 19.4188% |
| 20 | [0.3, 0.4) | 294 | 33.7170% | 24.8299% |
| 20 | [0.4, 0.5) | 66 | 44.4589% | 33.3333% |
| 20 | [0.5, 0.6) | 24 | 53.5939% | 33.3333% |
| 30 | [0.0, 0.1) | 800 | 5.0526% | 6.5000% |
| 30 | [0.1, 0.2) | 802 | 14.4002% | 8.9776% |
| 30 | [0.2, 0.3) | 292 | 23.6967% | 18.8356% |
| 30 | [0.3, 0.4) | 72 | 34.5020% | 16.6667% |
| 30 | [0.4, 0.5) | 55 | 44.9946% | 9.0909% |
| 30 | [0.5, 0.6) | 30 | 54.0462% | 13.3333% |
| 30 | [0.6, 0.7) | 17 | 65.6347% | 41.1765% |
| 30 | [0.7, 0.8) | 17 | 75.3022% | 29.4118% |
| 30 | [0.8, 0.9) | 10 | 84.6614% | 60.0000% |
| 30 | [0.9, 1.0) | 2 | 93.9213% | 100.0000% |

## Causality and leakage audit

- Past-only features: `True`
- Future used only in separate labels: `True`
- Complete-date split: `True`
- Purge bars per ETF: `30`
- Maximum active horizon: `30`
- Train/test overlapping same-day ETF windows: `False`
- OOS window previously reused: `True`

Residual risk remains: vendor bar timestamps and historical symbol availability are trusted as supplied, and this repeatedly used 2026-05-21 through 2026-06-18 window is not a clean final OOS test.

## Mechanical trade-frequency check

- Trading or gating applied: `False`
- Equal forecast rows across variants: `True`

Metric changes cannot be attributed to suppressing trades because all variants forecast the same rows and Phase 1 executes no trade.

## Fold stability

| Horizon | Joint Brier/AUC improving folds | Required | Stable |
|---:|---:|---:|---|
| 5 | 6/6 | 4 | True |
| 10 | 4/6 | 4 | True |
| 20 | 2/6 | 4 | False |
| 30 | 3/6 | 4 | False |

## Skipped

- `horizon_60`: A 60-bar horizon is 300 minutes at 5-minute frequency and crosses the 240-minute A-share session; Phase 1 leaves it null.
- `deep_lppls`: outside preregistered Phase 1 scope
- `koopman_dmd`: outside preregistered Phase 1 scope
- `hawkes`: outside preregistered Phase 1 scope
- `soc_avalanche`: outside preregistered Phase 1 scope
- `surrogate_ml`: outside preregistered Phase 1 scope
- `online_inference`: outside preregistered Phase 1 scope
- `live_trading_integration`: outside preregistered Phase 1 scope

## Phase 2 review

HMM + EWS does not achieve joint Brier/AUC improvement in at least two-thirds of folds on at least three horizons. The evidence supports retaining EWS as a shadow hypothesis, but does not justify entering LPPLS/Koopman Phase 2 yet.

This is a conservative post-run research review, not a preregistered model-selection or production-promotion gate.

No production recommendation is made.
