# Decision Score Daily Statistical Review

Status: `diagnostic_only / no automatic weight changes`

- as_of_date: 2026-06-22
- records: 1052; completed BUY-direction outcomes: 827; independent days: 1
- decision types: `{"BUY": 1, "BUY_CANDIDATE": 929, "HOLD": 118, "SELL": 4}`
- ledger types: `{"no_trade_buy_candidate": 929, "planned_order_decision": 5, "position_hold_decision": 118}`
- High/low comparisons are paired within the same market day; uncertainty resamples trading days.
- Holm correction controls the eight simultaneous score tests.

| score part | high n | low n | paired days | high mean | low mean | paired high-low | 95% CI | Holm p | diagnostic flags | assessment |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| total_score | 0 | 827 | 0 | - | 1.186% | - | [-, -] | - | low_not_weak | insufficient_forward_sample |
| market_regime_score | 491 | 0 | 0 | 0.714% | - | - | [-, -] | - | - | insufficient_forward_sample |
| relative_strength_score | 814 | 0 | 0 | 1.200% | - | - | [-, -] | - | - | insufficient_forward_sample |
| liquidity_score | 201 | 464 | 1 | 0.876% | 1.329% | -0.453% | [-, -] | - | high_not_strong, low_not_weak | insufficient_forward_sample |
| entry_quality_score | 380 | 424 | 1 | 1.528% | 0.851% | 0.677% | [-, -] | - | low_not_weak | insufficient_forward_sample |
| execution_score | 1 | 0 | 0 | 3.100% | - | - | [-, -] | - | - | insufficient_forward_sample |
| counterfactual_score | 814 | 0 | 0 | 1.200% | - | - | [-, -] | - | - | insufficient_forward_sample |
| risk_penalty | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |

## Adjustment audit

No score component is currently eligible for statistical adjustment.

Any future adjustment needs >=40 independent days, adequate high/low groups, Holm-adjusted evidence, and a separate OOS validation. Nothing is auto-applied.
