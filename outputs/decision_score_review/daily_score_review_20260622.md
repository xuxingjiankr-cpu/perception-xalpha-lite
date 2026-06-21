# Decision Score Daily Statistical Review

Status: `diagnostic_only / no automatic weight changes`

- as_of_date: 2026-06-22
- records: 0; completed BUY-direction outcomes: 0; independent days: 0
- decision types: `{}`
- ledger types: `{}`
- High/low comparisons are paired within the same market day; uncertainty resamples trading days.
- Holm correction controls the eight simultaneous score tests.

| score part | high n | low n | paired days | high mean | low mean | paired high-low | 95% CI | Holm p | diagnostic flags | assessment |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| total_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| market_regime_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| relative_strength_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| liquidity_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| entry_quality_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| execution_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| counterfactual_score | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |
| risk_penalty | 0 | 0 | 0 | - | - | - | [-, -] | - | - | insufficient_forward_sample |

## Adjustment audit

No score component is currently eligible for statistical adjustment.

Any future adjustment needs >=40 independent days, adequate high/low groups, Holm-adjusted evidence, and a separate OOS validation. Nothing is auto-applied.
