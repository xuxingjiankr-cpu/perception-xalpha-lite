# i03 OOS Variance Mechanism — Group Ablation

> CONTAMINATED WARNING: patched execution, but legacy Yahoo60 universe remains full-day-turnover contaminated.

Observed OOS 2026-05-21..06-18; diagnostic only.

| config | daily std | worst day | total PnL | Sharpe | std reduction | worst-day improvement |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4158.68 | -7184.80 | 13333.86 | 2.4237 | +0.00% | +0.00 |
| i03 | 3803.81 | -6847.50 | 16803.07 | 3.3393 | +8.53% | +337.30 |
| L1_risk_per_trade | 4006.92 | -6847.50 | 12853.95 | 2.425 | +3.65% | +337.30 |
| L2_earlier_profit | 4283.09 | -7184.80 | 15979.54 | 2.8203 | -2.99% | +0.00 |
| L3_faster_loss | 4158.68 | -7184.80 | 13333.86 | 2.4237 | +0.00% | +0.00 |
| L4_stress_selectivity | 3907.25 | -7184.80 | 14109.55 | 2.7297 | +6.05% | +0.00 |
| minimal_exploratory | 3762.71 | -6847.50 | 13624.13 | 2.7371 | +9.52% | +337.30 |

Driver groups (exploratory): `['L4_stress_selectivity', 'L1_risk_per_trade']`

Minimal config: `C:\Users\XU XINGJIAN\Documents\Codex\outputs\t0_replay\i03_ablation_minimal_runtime.json`

## Hard decision

exploratory only; validate on new post-2026-06-18 prospective data before shadow or live use.
