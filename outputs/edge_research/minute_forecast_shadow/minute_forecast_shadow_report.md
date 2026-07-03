# Five-Minute ETF Forecast Shadow Replay

Status: `diagnostic_only / frozen model / reused OOS / no live change`

- Train: 2026-03-23 to 2026-05-20.
- Test: 2026-05-21 to 2026-06-18.
- Frozen training-only universe: 50 ETFs.
- Every completed five-minute bar is scored; entry is the next observed bar and round-trip cost is 12 bps.

| Horizon | OOS samples | AUC | Brier / prior | Selected | Avg net trade | Policy return | Sharpe |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5m | 27819 | 0.8244413442427662 | 0.0811 / 0.0864 | 0 | n/a | 0.00% | n/a |
| 15m | 24419 | 0.7948872954158175 | 0.1032 / 0.1133 | 5 | -0.003012856951828689 | -0.31% | -4.252526381596193 |
| 30m | 19480 | 0.7714252679468449 | 0.1182 / 0.1300 | 1 | -0.008590746141365367 | -0.14% | -3.464101615137755 |

## Forward feature readiness

- Complete joint L2 + IOPV days: 5 / 20.
- L2 and IOPV are excluded from model fitting until the fixed day minimum is reached.

## Verdict

`no_validated_minute_forecast_edge`

No fixed horizon simultaneously beat the probability prior and produced positive cost-adjusted shadow returns.

A model must beat the training-prior probability forecast and produce positive cost-adjusted returns. Losing less than momentum or holding cash is not an edge.

## Limitations

- Historical observations are five-minute bars, not true one-minute bars.
- The 2026-05-21 to 2026-06-18 test window has already been inspected by other research and is not fresh OOS evidence.
- Yahoo history has no real historical order book, bid/ask or IOPV; 12 bps is a fixed execution-cost proxy.
- Current ETF membership can retain survivorship bias.
- The historical universe is selected on the complete training window and cannot evaluate newly listed or delisted products.
- Probability observations within the same timestamp and trading day are cross-sectionally dependent, so row-level AUC overstates effective sample size.
- Overlapping-horizon shadow returns use fixed equal tranches and are an approximation to concurrent portfolio accounting.
- L2 and IOPV histories currently have too few complete days and are deliberately excluded from model fitting.

This result cannot alter live entries, exits, sizing, overlays, orders or execution locks.
