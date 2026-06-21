# Decision Probability V1 — May Train / June Diagnostic Test

Status: `diagnostic_only / forward_shadow_not_promotable`

## Preregistered model

- BUY only; outcome = next-snapshot-to-close gross return minus 14bps > 0.
- Beta(2,2)-smoothed, day-balanced base rate; one total-score log-odds slope.
- Fixed L2=10. No component refit, grid search, isotonic fit or threshold search.
- The score likelihood term is one combined evidence term, so correlated sub-scores are not multiplied again.
- `tradeGateEnabled=false`; `positionSizingEnabled=false`; action is always record-only.

## Model parameters

- prior probability: 38.0174%
- total-score center / scale: 63.2045 / 7.7685
- regularized log-odds slope: +0.206422
- model confidence (independent days n/(n+50)): 26.4706%
- expected gross win / loss / cost: 1.3991% / 0.8476% / 0.1400%

## Fixed June diagnostic

- sample: 60 BUY outcomes / 13 days

| forecast | Brier ↓ | LogLoss ↓ | AUC ↑ | ECE ↓ | mean p | actual win rate |
|---|---:|---:|---:|---:|---:|---:|
| constant May prior | 0.264358 | 0.722721 | 0.500000 | 0.119826 | 0.380174 | 0.500000 |
| neutral 50% reference | 0.250000 | 0.693147 | 0.500000 | 0.000000 | 0.500000 | 0.500000 |
| one-slope posterior | 0.257845 | 0.710769 | 0.608889 | 0.111856 | 0.388144 | 0.500000 |

### Reliability bins

| predicted interval | n | mean predicted | actual rate |
|---|---:|---:|---:|
| [0.2, 0.4) | 30 | 34.7108% | 36.6667% |
| [0.4, 0.6) | 30 | 42.9180% | 63.3333% |

## Verdict

- test-day gate (>=20): `False`
- calibration beats constant prior on both Brier and LogLoss: `True`
- calibration beats neutral 50% on both Brier and LogLoss: `False`
- promotion_allowed: `false`
- AUC shows a weak ranking hint, but predicted probabilities are materially too low in June and do not beat the neutral 50% reference on proper scoring rules.
- June has only 13 independent days and was already inspected during score research. It is not a clean confirmation.
- The checked-in parameters are a preregistered forecast for data from 2026-06-22 onward. Refit is manual only after >=50 BUY outcomes and >=20 independent days.
- Regime priors, four evidence-group LRs, EV gating and Bayesian position sizing remain deferred; fitting them now would be small-sample overfit.
