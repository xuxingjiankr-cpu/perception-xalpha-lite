# DSI-0008: High-vs-Low BUY Score Separation

Status: `diagnostic_only / not_validated`

Fixed June point-in-time test; BUY only; 14bps deducted; high >=71, low <=65.

| group | decisions | days | mean net | median net | win rate | mean MAE | mean MFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| high | 28 | 8 | 0.2235% | 0.4010% | 60.71% | -0.6689% | 1.1711% |
| low | 27 | 12 | -0.0423% | -0.1240% | 40.74% | -0.2774% | 0.5177% |

- pooled high-minus-low: 0.2658%
- day-balanced high mean: 0.2767%; low mean: -0.0533%
- same-day paired difference (7 days): -0.0448%
- day-cluster bootstrap 95% CI: [-0.6554%, 1.0948%]
- one-sided probability difference <= 0: 26.51%

## Verdict

- pooled_high_better: `True`
- cluster_ci_excludes_zero: `False`
- same_day_paired_high_better: `False`
- validated: `False`

High scores look better in the pooled sample, but not after controlling for trading day. The score currently appears to capture favorable market days more than superior same-day ETF selection. Keep it shadow-only and do not gate orders.
