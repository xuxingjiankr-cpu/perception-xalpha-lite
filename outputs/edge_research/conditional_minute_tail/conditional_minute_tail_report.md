# Conditional Minute Tail-Risk Research

Status: `diagnostic_only / reused OOS / no live change`

The model treats bars as conditional innovations, not unconditional IID rows. Dependence is retained through a common market factor, a filtered HMM state distribution and lagged EWMA idiosyncratic variance.

| Loss in next 5m | Events | Conditional / EWMA / static AUC | Conditional / EWMA / static Brier | Daily Brier improvement |
|---|---:|---:|---:|---:|
| 0.20% | 2156 | 0.8837 / 0.8827 / 0.8716 | 0.042974 / 0.042953 / 0.043851 | 0.000919 |
| 0.30% | 1167 | 0.8801 / 0.8807 / 0.8698 | 0.025222 / 0.025179 / 0.025717 | 0.000522 |
| 0.50% | 314 | 0.9150 / 0.9175 / 0.9020 | 0.007270 / 0.007258 / 0.007618 | 0.000366 |
| 1.00% | 32 | 0.9200 / 0.9403 / 0.9193 | 0.000880 / 0.000880 / 0.000791 | -0.000083 |

## Dependence audit

- Median lag-1 squared-return correlation: 0.2252.
- After factor removal and causal variance standardisation: 0.1331.
- Mean absolute cross-sectional correlation: 0.1274 raw versus 0.1268 after common-factor removal.

## Verdict

`conditional_dependence_risk_signal_not_supported`

The conditional model failed at least one sample, calibration, ranking or dependence-reduction gate.

The result is a risk ranking only. It does not show that avoiding high-risk bars improves cost-adjusted return.

## Limitations

- The observations are five-minute bars, not true one-minute quotes.
- The 2026-05-21 to 2026-06-18 window has already been inspected and is not fresh OOS evidence.
- The conditional Gaussian mixture is a model assumption, not a distribution-free probability guarantee.
- The common factor is contemporaneously observed only when updating the next forecast; each forecast is made before the current bar is observed.
- Row-level AUC and Brier observations are cross-sectionally dependent, so daily-block diagnostics are reported separately.
- The training-only universe retains possible survivorship bias.
- Historical Yahoo bars do not contain executable historical bid, ask, depth or IOPV.
- A tail-risk ranking can reduce exposure to volatile ETFs but cannot create positive expected return.

No live config, overlay, order path, execution lock or position sizing rule was changed.
