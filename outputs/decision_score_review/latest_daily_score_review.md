# Decision Score Daily Statistical Review

Status: `diagnostic_only / no automatic weight changes`

- as_of_date: 2026-06-26
- records: 1959; completed BUY-direction outcomes: 1359; independent days: 5
- decision types: `{"BUY": 5, "BUY_CANDIDATE": 1632, "HOLD": 312, "SELL": 10}`
- ledger types: `{"no_trade_buy_candidate": 1632, "planned_order_decision": 15, "position_hold_decision": 311, "snapshot_decision": 1}`
- High/low comparisons are paired within the same market day; uncertainty resamples trading days.
- Holm correction controls the eight simultaneous score tests.

## Fixed total-score buckets

| total_score bucket | outcomes | executed | candidates | win rate | avg net | median net | expectancy | avg MAE | avg MFE | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-40 | 317 | 0 | 317 | 37.539% | -0.248% | -0.175% | -0.248% | -0.930% | 0.623% | -3.384% |
| 40-60 | 968 | 2 | 966 | 70.868% | 0.818% | 0.531% | 0.818% | -0.616% | 1.182% | -3.380% |
| 60-75 | 73 | 2 | 71 | 47.945% | 0.194% | -0.027% | 0.194% | -0.876% | 0.719% | -3.206% |
| 75-90 | 1 | 1 | 0 | 0.000% | -0.550% | -0.550% | -0.550% | -0.512% | 0.820% | -0.550% |
| 90+ | 0 | 0 | 0 | - | - | - | - | - | - | - |

## High/low score-part tests

| score part | high n | low n | paired days | high mean | low mean | paired high-low | 95% CI | Holm p | diagnostic flags | assessment |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| total_score | 3 | 1353 | 2 | -1.351% | 0.545% | -0.545% | [-0.833%, -0.257%] | 0.005997 | high_not_strong, high_non_positive, low_not_weak | insufficient_forward_sample |
| market_regime_score | 657 | 193 | 3 | 0.415% | -0.621% | -0.623% | [-0.947%, -0.189%] | 0.005997 | high_not_strong | insufficient_forward_sample |
| relative_strength_score | 974 | 210 | 4 | 0.925% | -0.590% | -0.051% | [-0.184%, 0.082%] | 1.000000 | high_not_strong | insufficient_forward_sample |
| liquidity_score | 459 | 709 | 5 | 0.100% | 0.737% | -0.116% | [-0.323%, 0.087%] | 0.879560 | high_not_strong, low_not_weak | insufficient_forward_sample |
| entry_quality_score | 401 | 916 | 4 | 1.418% | 0.169% | -0.153% | [-0.778%, 0.430%] | 1.000000 | high_not_strong, low_not_weak | insufficient_forward_sample |
| execution_score | 5 | 0 | 0 | -0.397% | - | - | [-, -] | - | high_non_positive | insufficient_forward_sample |
| counterfactual_score | 974 | 210 | 4 | 0.925% | -0.590% | -0.051% | [-0.184%, 0.082%] | 1.000000 | high_not_strong | insufficient_forward_sample |
| risk_penalty | 365 | 0 | 0 | -0.483% | - | - | [-, -] | - | high_non_positive | insufficient_forward_sample |

## Adjustment audit

No score component is currently eligible for statistical adjustment.

Any future adjustment needs >=40 independent days, adequate high/low groups, Holm-adjusted evidence, and a separate OOS validation. Nothing is auto-applied.
