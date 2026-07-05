# Koopman paper-only exception audit

- Run ID: `koopman_exception_20260705_v1`
- Status: `diagnostic_only`
- Same-sample rows: 4,396
- Coverage exception used: `True`
- 10-bar exception used: `True`
- High-risk sample exception used: `True`

| variant | Brier | LogLoss | AUC | ECE | high-risk n |
|---|---:|---:|---:|---:|---:|
| HMM+EWS+LPPLS | 0.028836 | 0.133799 | 0.607983 | 0.003636 | 7 |
| +Koopman/DMD | 0.028821 | 0.133671 | 0.612671 | 0.003621 | 7 |

- Improved metrics: 4/4
- Improving folds: 4/7
- Improving months: 6/13
- Improving ETF categories: 4
- Brier delta candidate-baseline: -0.00001526
- Cluster bootstrap CI: [-0.00007777, 0.00004307]

- Non-waivable gates passed: `False`
- Paper risk-veto integration allowed: `False`
- Main paper config modified: `False`

Even after waiving coverage, 10-bar and high-risk sample gates, Koopman does not pass the non-waivable predictive gates and is not integrated into paper trading.
