# Kronos ETF zero-shot shadow study

## Status and purpose

This study is `research_only`, `diagnostic_only` and `shadow_only`. It asks
whether the open-source Kronos candlestick model supplies incremental
out-of-time information for liquid A-share ETF five-minute bars. It is not a
trading strategy and cannot write orders, positions, production probability
artifacts, overlays, risk gates or `build_decision()`.

Kronos is used without fine-tuning. The upstream model and tokenizer, sampling
parameters, universe, decision time, horizons, cost and date splits are frozen
in `configs/research/kronos_etf_shadow_preregistered.json`.

## Causal observation and label

At a configured decision time, the input ends at the completed five-minute
bar. The 400-bar OHLCVA context may include prior sessions. A hypothetical
entry occurs only at the next bar's open. For horizon `h`, the label is:

`close[t+h] / open[t+1] - 1 - 15.5 bps round-trip cost > 0`

The label and realized return live only in the offline evaluation table.
Future rows are never supplied to the predictor or baseline features. All
horizons must finish in the same A-share session.

## Frozen experiment

- fixed universe: eight liquid cross-border/commodity ETFs with local OHLCVA;
- baseline fit: 2024-07-01 through 2024-12-31;
- probability calibration and ensemble selection: 2025-01-01 through
  2025-06-30;
- untouched OOS: 2025-07-01 through 2026-03-20;
- one decision per day at 11:00 to bound compute and prevent timestamp fishing;
- Kronos-small, tokenizer-base, 400-bar context;
- horizons: 1, 3, 6 and 12 bars;
- ten stochastic paths, temperature 0.6 and top-p 0.9.

The causal baseline is a regularized logistic model using only past OHLCVA
returns, range, volatility, volume and session progress. Kronos probabilities
are the fraction of sampled paths whose predicted next-open-to-horizon-close
return clears the frozen cost. Platt calibration and a five-value ensemble
weight grid are fit only in the calibration period, then frozen for OOS.

## Outputs

Each run writes to
`outputs/edge_research/kronos_etf_shadow/<run_id>/`:

- `data_audit.json`;
- `forecast_rows.jsonl` when model inference is requested;
- `metrics.json`;
- `probability_buckets.json`;
- `report.md`;
- `run_manifest.json`.

`--audit-only` performs the full data/causality audit without loading model
weights. `--smoke` bounds inference and produces a non-conclusive integration
test. A smoke result can never be promoted or described as edge.

## Decision rule

No production discussion is allowed unless the frozen OOS contains at least
60 independent trading days, the high-probability buckets contain at least
100 rows, Kronos or the frozen blend improves most of Brier, LogLoss, AUC and
ECE over the baseline at both 3 and 6 bars, ranking and net-return monotonicity
persist by month and symbol, and the gain is not explained merely by fewer
selected rows.

Passing this gate would authorize a separate preregistered forward-shadow
discussion only. It would not authorize BUY/SELL gating or position sizing.
