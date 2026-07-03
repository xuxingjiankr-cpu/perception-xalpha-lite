# Chebyshev/Cantelli and Chernoff Tail-Risk Shadow Replay

Status: `diagnostic_only / reused contaminated replay / no live change`

- Decision timing: each session's exposure uses only the previous 20 strategy-PnL days.
- Risk budget: upper-bound P(daily loss > 1.0%) <= 10%.
- This tests exposure scaling, not entry/exit alpha.

| Method | Return | Daily std | Worst day | Max DD | Avg exposure | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3.58% | 0.41% | -0.72% | -1.21% | 100.00% | 3.4280307660216196 |
| cantelli | 3.03% | 0.30% | -0.54% | -0.91% | 69.38% | 3.933381778506124 |
| chernoff_ucb | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | n/a |
| best_valid_bound | 3.03% | 0.30% | -0.54% | -0.91% | 69.38% | 3.933381778506124 |

## Interpretation

- Cantelli reduced daily standard deviation by 26.2%, improved the worst day by 0.18%, and retained 84.7% of return.
- The finite-sample Chernoff-Hoeffding UCB selected zero exposure throughout this short sample. It is mathematically conservative but not operationally useful here.
- `best_valid_bound` equals Cantelli because its bound dominates the uninformative Chernoff UCB in this sample.
- The apparent drawdown improvement is largely mechanical exposure reduction; it is not proof that a tail event was predicted.

## Verdict

`historical_risk_shape_supported_forward_shadow_required`

The fixed Cantelli budget preserved at least 80% of historical return while reducing dispersion and drawdown, but reused and look-ahead-contaminated data prohibit deployment.

## Limitations

- The 60-day replay and its test window have already been inspected; this is not clean OOS evidence.
- The historical universe used same-day final turnover and is contaminated by liquidity look-ahead.
- Scaling an already known daily PnL mechanically reduces both gains and losses; it does not prove tail-event predictability.
- Cantelli uses only mean and variance and is generally loose.
- The finite-sample Chernoff calculation additionally assumes independent observations and true daily returns bounded within plus or minus 5%.
- Twenty rolling observations are too few for stable tail inference.
- The replay summaries come from different historical replay stages and are suitable only for a diagnostic risk-shape test.

No live config, order path, sizing rule, overlay or execution lock was changed.
