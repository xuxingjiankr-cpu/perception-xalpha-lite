# Calibration Bins — DCAL-1.0.1

No Platt/isotonic refit is applied. These are forward diagnostics only.

- Brier: 0.342138
- LogLoss: 0.896007
- AUC: 0.624662
- ECE / MCE: 0.341428 / 0.429012

| probability_bin | count | avgPredProb | actualWinRate | calibrationError | Brier | LogLoss |
|---|---:|---:|---:|---:|---:|---:|
| 0.00—0.20 | 128 | 17.41% | 34.38% | 16.97% | 0.257973 | 0.740659 |
| 0.20—0.30 | 707 | 25.92% | 60.11% | 34.19% | 0.351455 | 0.918701 |
| 0.30—0.40 | 518 | 33.40% | 71.62% | 38.22% | 0.352048 | 0.907305 |
| 0.40—0.50 | 6 | 42.90% | 0.00% | -42.90% | 0.184241 | 0.560683 |

## Raw probability histogram

DCAL-1.0.1 applies no new calibration transform, so raw and calibrated probabilities are currently identical by design.

| probability_bin | count | share |
|---|---:|---:|
| 0.00—0.20 | 128 | 9.42% |
| 0.20—0.30 | 707 | 52.02% |
| 0.30—0.40 | 518 | 38.12% |
| 0.40—0.50 | 6 | 0.44% |
