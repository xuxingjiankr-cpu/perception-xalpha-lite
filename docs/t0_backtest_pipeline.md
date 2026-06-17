# T0 ETF Layered Backtest Pipeline

Paper-trading research only. This pipeline does not call the broker API, submit
orders, change the three execution locks, or update the agent-facing evolution
overlay.

## Profiles

- `smoke_test`: one day, baseline only, lightweight output.
- `fast_screening`: three complete days, fixed candidates, one-day first stage,
  then the best three plus baseline on all selected days.
- `medium_screening`: five to ten complete days with the same two-stage search.
- `full_replay`: at least ten complete days; requires a promoted, frozen parameter
  artifact and writes full replay details.
- `oos_replay`: at least five complete out-of-sample days; requires an explicit
  non-overlapping date range and promoted, frozen parameters.

Data completeness requires both intraday coverage and cross-sectional breadth.
This prevents a day with 240 timestamps but only a handful of ETFs from entering
medium, full, or OOS validation.

## Commands

```powershell
$env:PYTHONIOENCODING='utf-8'

py -3.13 scripts/run_t0_backtest_pipeline.py `
  --profile smoke_test `
  --config outputs/t0_replay/config_replay_all_etf_202606.json `
  --quotes outputs/t0_replay/replay_all_etf_20260601_20260617.jsonl `
  --start-date 2026-06-17 --end-date 2026-06-17 --use-cache

py -3.13 scripts/run_t0_backtest_pipeline.py `
  --profile fast_screening `
  --config outputs/t0_replay/config_replay_all_etf_202606.json `
  --quotes outputs/t0_replay/replay_all_etf_20260601_20260617.jsonl `
  --start-date 2026-06-10 --end-date 2026-06-15 --use-cache

py -3.13 scripts/run_t0_backtest_pipeline.py `
  --profile oos_replay `
  --config outputs/t0_replay/config_replay_all_etf_202606.json `
  --quotes <new_complete_quote_source> `
  --start-date <oos_start> --end-date <oos_end> `
  --locked-parameters <medium_run>/promoted_params.json --use-cache
```

## Outputs

- Per-run: `outputs/t0_backtest_pipeline/<profile>/<run_id>/`
- Registry: `outputs/experiments/experiment_registry.csv`
- Reusable result cache: `outputs/t0_backtest_pipeline/cache/`
- Data completeness audit cache: `outputs/t0_backtest_pipeline/data_audit_cache/`

Each run records input/code/parameter/cost hashes, runtime breakdown, cache use,
validation failures, all tested candidates, and 6/12/20 bps round-trip cost
sensitivity. Failed and rejected experiments remain in the registry.

## Known Limitation

The current replay submits and locally fills a marketable limit order in the same
snapshot event after signal construction. It does not access a future snapshot,
but it is optimistic relative to queue-aware execution. Full/OOS promotion still
requires an independent execution-model validation. Backtrader may later be used
as an external event-engine cross-check; it is not a replacement for the current
A-share ETF state machine.
