# OOS Variance / Drawdown Validation

> CONTAMINATED WARNING: legacy Yahoo60 universe used same-day final turnover. Diagnostic only; not edge evidence.

Paper-only diagnostic. Frozen train: 2026-03-23..05-20; untouched OOS: 2026-05-21..06-18.

| config | worst day | daily std | max cumulative DD | total PnL | winning days | net PnL @12bps | net std @12bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | -6868.60 | 4324.56 | -12923.80 | 13822.80 | 12/21 | 2160.56 | 4373.33 |
| i01 | -7666.00 | 4440.01 | -8431.00 | 14978.60 | 12/21 | 4483.13 | 4425.31 |
| i02 | -7915.70 | 3869.64 | -14612.70 | 9505.20 | 11/21 | -2301.26 | 3934.48 |
| i03 | -4958.50 | 3273.35 | -10888.40 | 12923.90 | 11/21 | 2099.66 | 3325.31 |
| i05 | -7308.40 | 3566.21 | -13690.90 | 344.30 | 9/21 | -12277.99 | 3609.85 |

- Status: `oos_lower_variance_supported`
- Passing: `['i03']`

- i01: std -2.67% reduction; worst-day improvement -797.40; defensible pass=False
- i02: std +10.52% reduction; worst-day improvement -1047.10; defensible pass=False
- i03: std +24.31% reduction; worst-day improvement +1910.10; defensible pass=True
- i05: std +17.54% reduction; worst-day improvement -439.80; defensible pass=False

## Decision

freeze the lower-variance hypothesis; optimize future research for risk-adjusted return/drawdown, then validate on new post-2026-06-18 data.

A new optimizer run is intentionally not started: the OOS period has now informed objective selection. Any new risk-adjusted candidate requires prospective data after 2026-06-18.
