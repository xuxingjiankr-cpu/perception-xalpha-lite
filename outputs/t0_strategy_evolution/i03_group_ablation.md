# i03 OOS Variance Mechanism — Group Ablation

Observed OOS 2026-05-21..06-18; diagnostic only.

| config | daily std | worst day | total PnL | Sharpe | std reduction | worst-day improvement |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4324.56 | -6868.60 | 13822.80 | 2.4162 | +0.00% | +0.00 |
| i03 | 3273.35 | -4958.50 | 12923.90 | 2.9846 | +24.31% | +1910.10 |
| L1_risk_per_trade | 4215.23 | -6623.40 | 13825.60 | 2.4794 | +2.53% | +245.20 |
| L2_earlier_profit | 4324.93 | -6869.50 | 14201.20 | 2.4822 | -0.01% | -0.90 |
| L3_faster_loss | 3768.86 | -6852.10 | 10339.20 | 2.0737 | +12.85% | +16.50 |
| L4_stress_selectivity | 4182.66 | -5463.60 | 18548.30 | 3.3522 | +3.28% | +1405.00 |
| minimal_exploratory | 3374.90 | -5183.60 | 12263.80 | 2.7469 | +21.96% | +1685.00 |

Driver groups (exploratory): `['L3_faster_loss', 'L4_stress_selectivity']`

Minimal config: `C:\Users\XU XINGJIAN\Documents\Codex\outputs\t0_strategy_evolution\candidate_configs\oos_i03_minimal_risk_exploratory.json`

## Hard decision

exploratory only; validate on new post-2026-06-18 prospective data before shadow or live use.
