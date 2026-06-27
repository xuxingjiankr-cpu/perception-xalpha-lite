# Bayesian Prior Registry — DBAYES-1.0.1

Status: `research_only`; allowed_for_trade_gate: `false`

Beta prior: alpha=2.0, beta=2.0; confidence=n/(n+50).
Overall: n=1359, days=5, posterior=61.78%, confidence=96.45%.
Configured LR values are config-only; none are learned automatically.

## market_regime

| group | n | days | posterior | confidence |
|---|---:|---:|---:|---:|
| neutral | 1359 | 5 | 61.78% | 96.45% |

## strategy_type

| group | n | days | posterior | confidence |
|---|---:|---:|---:|---:|
| breakout | 55 | 4 | 54.24% | 52.38% |
| exit | 72 | 1 | 97.37% | 59.02% |
| momentum_trend | 24 | 2 | 89.29% | 32.43% |
| unclassified | 1208 | 5 | 59.16% | 96.03% |

## holding_horizon

| group | n | days | posterior | confidence |
|---|---:|---:|---:|---:|
| intraday_to_close | 1359 | 5 | 61.78% | 96.45% |

## symbol_group

| group | n | days | posterior | confidence |
|---|---:|---:|---:|---:|
| broad_index | 40 | 1 | 79.55% | 44.44% |
| cross_border | 366 | 5 | 35.95% | 87.98% |
| financial_defensive | 81 | 1 | 97.65% | 61.83% |
| resource_commodity | 165 | 5 | 56.21% | 76.74% |
| sector_other | 413 | 2 | 72.66% | 89.20% |
| technology_growth | 294 | 2 | 68.12% | 85.47% |

## signal_direction

| group | n | days | posterior | confidence |
|---|---:|---:|---:|---:|
| BUY | 1359 | 5 | 61.78% | 96.45% |
