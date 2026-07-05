# Koopman one-minute direction study

- Run ID: `koopman_minute_direction_20260705_v1`
- Status: `diagnostic_only`
- Data: 2026-02-06 through 2026-07-03
- Universe / trading days / raw rows: 18 / 95 / 408,480
- Koopman direction rows: 357,420 (97.22% of post-warm-up opportunities)
- Intraday availability: 10:00 through 14:59, recomputed every minute

The target is next-minute up/down direction. Exact zero returns are retained in the separate label table as neutral and excluded from binary fitting. Features use only completed same-session bars.

| variant | n | Brier | LogLoss | AUC | ECE | accuracy@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| causal_baseline | 122,738 | 0.248069 | 0.689648 | 0.562955 | 0.020972 | 54.8494% |
| baseline_plus_koopman | 122,738 | 0.248040 | 0.689579 | 0.563148 | 0.020541 | 54.8290% |

- Improved metrics: 4/4
- Improving folds: 4/5
- Improving months: 3/4
- Improving ETF categories: 6
- Brier delta candidate-baseline: -0.00002939
- Trading-date cluster CI: [-0.00004653, -0.00001229]
- All preregistered gates passed: `True`
- Paper or online integration allowed: `False`

The fixed per-minute Koopman direction covariates pass the historical evidence gate. They remain research-only pending a separate immutable shadow artifact and fresh forward validation.

This is a directional context model, not a turning-probability model and not an order generator. No trading configuration, SELL path, sizing rule, risk gate, overlay, or execution lock changed.
