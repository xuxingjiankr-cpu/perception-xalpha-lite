# Frozen Decision-Score Pseudo-Forward Report

Trust level: `diagnostic_only`
Sample type: `pseudo_forward` (chronological bars, but strategy/scorer post-date the sample)
Weights refit allowed: `false`

## Frozen inputs

- source commit: `3ddc45c665a1d10bc724b19debff0af894ece631`
- scorer SHA256: `9d64efd26e28ce8a97716f281947dcb7aafdedc801e104d92e08e0e598176a38`
- config SHA256: `13cfddba50162e4a593b9fc5f72c0bc0147bcb8b5b9bd2c94ae48088cf727ba7`
- requested window: 2026-05-06..2026-06-18
- replayed trading days: 18
- confirmed T0 universe records: 131

## Portfolio replay

- gross realized P&L before path costs: 10,440.50
- total equity P&L: -2,344.05
- ending equity: 997,655.95
- max drawdown: -3.4220%
- completed trades: 28
- transaction cost: 12,784.55
- winning days: 5/18
- median daily net P&L: 0.00
- result trust from replay: `diagnostic_only`

## Frozen-score diagnostic

- recorded decisions: 771 ({'HOLD': 673, 'BUY': 60, 'SELL': 38})
- directional BUY/SELL outcomes: 96 across 13 days
- total-score/return Pearson correlation: -0.0558

| score bucket | directional outcomes | mean return |
|---|---:|---:|
| A | 0 | - |
| B | 0 | - |
| C | 33 | 0.3508% |
| D | 20 | 0.0669% |
| E | 43 | 0.3608% |

## Data coverage

| date | replay rows | point-in-time eligible codes |
|---|---:|---:|
| 2026-05-19 | 41 | 1 |
| 2026-05-20 | 2 | 1 |
| 2026-05-26 | 8 | 1 |
| 2026-05-29 | 48 | 1 |
| 2026-06-01 | 3633 | 90 |
| 2026-06-02 | 3721 | 88 |
| 2026-06-03 | 3416 | 83 |
| 2026-06-04 | 3324 | 78 |
| 2026-06-05 | 3290 | 78 |
| 2026-06-08 | 3344 | 80 |
| 2026-06-09 | 3474 | 81 |
| 2026-06-10 | 3498 | 81 |
| 2026-06-11 | 3239 | 79 |
| 2026-06-12 | 3599 | 91 |
| 2026-06-15 | 3941 | 94 |
| 2026-06-16 | 3761 | 90 |
| 2026-06-17 | 3577 | 86 |
| 2026-06-18 | 3960 | 95 |

Source builds:

- 2026-05-06..2026-05-31: loaded 20/131 symbols, rows=99
- 2026-06-01..2026-06-18: loaded 131/131 symbols, rows=49777

## Limitations

- May source coverage is incomplete and expands through the month; missing instruments fail closed.
- The historical order book is synthetic, so fill probability and queue position are not observed.
- The current strategy and scorer were created after this sample existed; chronological replay removes row look-ahead but not researcher-selection bias.
- The T0 confirmed pool is a current master, not a point-in-time May master; survivor bias remains.
- HOLD/SKIP are excluded from directional score claims; they are not treated as synthetic long/short trades.
- No parameter or score weight may be changed from this result. Only post-2026-06-21 real forward data can support refitting.

## Verdict

This run is useful for checking chronological decisions, cash/position accounting and whether the frozen score is directionally coherent. It is not clean alpha evidence and cannot promote or reweight the live strategy.
