# Forward Shadow Probability Ledger Summary

Status: `research_only / trade_invalid_probability`

- effective_from: 2026-06-22
- completed BUY-direction forecasts: 1359 across 5 days
- planned BUY / no-trade counterfactual: 5 / 1354
- final decisions / no-trade candidates recorded: 327 / 1632
- candidate coverage: 100.00%
- incomplete fixed horizons: 216

No candidate row is treated as a fill. Candidate outcomes are explicitly next-snapshot-to-close counterfactual BUY returns.
No probability field can gate or size an order.

## Daily diagnostics

| date | n | mean p | actual rate | Brier | LogLoss | AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-22 | 827 | 30.16% | 80.65% | 0.413650 | 1.045767 | 0.482323 | 0.504902 |
| 2026-06-23 | 258 | 22.10% | 20.93% | 0.182346 | 0.561934 | 0.270788 | 0.154167 |
| 2026-06-24 | 89 | 28.69% | 76.40% | 0.430878 | 1.098333 | 0.252801 | 0.486688 |
| 2026-06-25 | 95 | 25.59% | 14.74% | 0.143513 | 0.467089 | 0.441358 | 0.108513 |
| 2026-06-26 | 90 | 27.60% | 41.11% | 0.265003 | 0.730227 | 0.496175 | 0.192656 |
