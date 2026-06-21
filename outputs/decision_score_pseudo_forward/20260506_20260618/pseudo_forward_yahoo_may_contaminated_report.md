# Frozen Decision-Score Pseudo-Forward Report

Trust level: `contaminated`
Sample type: `pseudo_forward_with_contaminated_yahoo_may` (chronological bars, but strategy/scorer post-date the sample)
Weights refit allowed: `false`

## Frozen inputs

- source commit: `3ddc45c665a1d10bc724b19debff0af894ece631`
- scorer SHA256: `9d64efd26e28ce8a97716f281947dcb7aafdedc801e104d92e08e0e598176a38`
- config SHA256: `13cfddba50162e4a593b9fc5f72c0bc0147bcb8b5b9bd2c94ae48088cf727ba7`
- requested window: 2026-05-06..2026-06-18
- replayed trading days: 32
- confirmed T0 universe records: 131

## Portfolio replay

- gross realized P&L before path costs: 32,741.85
- total equity P&L: 6,485.42
- ending equity: 1,006,485.42
- max drawdown: -5.1069%
- completed trades: 60
- transaction cost: 26,256.43
- winning days: 15/32
- median daily net P&L: 0.00
- result trust from replay: `contaminated`

## Frozen-score diagnostic

- recorded decisions: 1536 ({'HOLD': 1302, 'BUY': 155, 'SELL': 79})
- directional BUY/SELL outcomes: 232 across 31 days
- total-score/return Pearson correlation: -0.0359

| score bucket | directional outcomes | mean return |
|---|---:|---:|
| A | 0 | - |
| B | 0 | - |
| C | 83 | 0.0611% |
| D | 58 | 0.0046% |
| E | 91 | 0.1471% |

## Data coverage

| date | replay rows | point-in-time eligible codes |
|---|---:|---:|
| 2026-05-06 | 3776 | 79 |
| 2026-05-07 | 3694 | 77 |
| 2026-05-08 | 3538 | 74 |
| 2026-05-11 | 3827 | 80 |
| 2026-05-12 | 3678 | 77 |
| 2026-05-13 | 3633 | 76 |
| 2026-05-14 | 3677 | 77 |
| 2026-05-15 | 3683 | 77 |
| 2026-05-18 | 3633 | 76 |
| 2026-05-19 | 3668 | 77 |
| 2026-05-20 | 3589 | 75 |
| 2026-05-21 | 3631 | 76 |
| 2026-05-22 | 3634 | 76 |
| 2026-05-25 | 2764 | 58 |
| 2026-05-26 | 3672 | 77 |
| 2026-05-27 | 3827 | 80 |
| 2026-05-28 | 3875 | 81 |
| 2026-05-29 | 3972 | 83 |
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

- 2026-05-06..2026-05-31: loaded 89/131 symbols, rows=65771
- 2026-06-01..2026-06-18: loaded 131/131 symbols, rows=49777

## Limitations

- May source coverage is incomplete and expands through the month; missing instruments fail closed.
- The historical order book is synthetic, so fill probability and queue position are not observed.
- The current strategy and scorer were created after this sample existed; chronological replay removes row look-ahead but not researcher-selection bias.
- The T0 confirmed pool is a current master, not a point-in-time May master; survivor bias remains.
- HOLD/SKIP are excluded from directional score claims; they are not treated as synthetic long/short trades.
- No parameter or score weight may be changed from this result. Only post-2026-06-21 real forward data can support refitting.

- Yahoo fallback used a final-day-turnover-filtered universe; this entire companion run is `contaminated`.
## Verdict

This run is useful for checking chronological decisions, cash/position accounting and whether the frozen score is directionally coherent. It is not clean alpha evidence and cannot promote or reweight the live strategy.
