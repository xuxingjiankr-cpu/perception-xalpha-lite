# Minute Forecast + HMM + Black-Litterman Fusion

Status: `diagnostic_only / fixed ablation / no live change`

- Common horizon: 30 minutes.
- Next-bar entry and 12 bps round-trip cost for every variant.
- Old MLP is intentionally not restacked; HMM supplies state and BL supplies shrinkage/allocation.

| Variant | Trades | Avg net trade | Net return | Sharpe | Worst day | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| new_linear | 1 | -0.008590746141365367 | -0.14% | -3.464101615137755 | -0.14% | -0.14% |
| new_plus_hmm_features | 1 | -0.008590746141365367 | -0.14% | -3.464101615137755 | -0.14% | -0.14% |
| new_plus_hmm_and_bl | 1 | -0.008590746141365367 | -0.02% | -3.464101615137755 | -0.02% | -0.02% |

## Verdict

`fusion_does_not_create_validated_edge`

No fixed fusion variant produced enough positive cost-adjusted trades with positive absolute portfolio return.

Lower loss caused only by lower exposure or fewer trades is not counted as alpha.

## Limitations

- The historical test window has already been reused and cannot support promotion.
- All variants use the same Yahoo five-minute bars and fixed 12 bps cost proxy.
- The HMM and linear predictors share price-derived information and are not independent alpha sources.
- Black-Litterman can stabilize views but cannot create expected return when forecasts lack edge.
- Sparse selections can appear to reduce loss simply by holding cash.
- Current product membership can retain survivorship bias.

No live config, order path, sizing rule, overlay or execution lock was changed.
