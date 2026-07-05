# Singularity Phase 2A data audit

- Status: `pass`
- Range: 2024-06-12 through 2026-07-03
- Trading days / symbols / rows: 500 / 18 / 425,760
- Maximum missing-bar fraction: 0.4032%
- Invalid OHLC / duplicates: 0 / 0

## Effective labels and independent events

| horizon | effective rows | positive rows | independent events |
|---:|---:|---:|---:|
| 5 | 283,840 | 10,579 | 6,180 |
| 10 | 239,490 | 14,528 | 5,607 |
| 20 | 150,790 | 13,200 | 4,002 |
| 30 | 62,090 | 6,041 | 2,320 |
| 60 | 0 | 0 | 0 |

## Feasible fixed windows

- LPPLS fixed windows: 44,350
- LPPLS trend-eligible windows: 921
- Equal trading-time DMD windows: 44,350

## Gate results

- `minimumSymbols`: `True`
- `minimumTradingDays`: `True`
- `maximumMissingBarFraction`: `True`
- `maximumInvalidOhlcRows`: `True`
- `minimumIndependentEvents5`: `True`
- `minimumIndependentEvents10`: `True`
- `minimumLpplsWindows`: `True`
- `minimumKoopmanWindows`: `True`

Raw bars have no adjustment factor. Overnight discontinuities are reported, but all model features and labels reset within each day.
Tick data are absent; L2 and IOPV histories are too short and are not model inputs.
