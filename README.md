# ETF Paper Trading Agent

This repository contains the local A-share ETF paper-trading agent used for competition experiments.

## Scope

- Paper trading only.
- Not live-ready.
- Not a formal strategy.
- Not an investment recommendation.
- Do not commit API keys, account snapshots, runtime logs, or broker output.

## Main Files

- `configs/t0_intraday_paper_agent.json` - T+0 intraday ETF agent configuration.
- `configs/etf_paper_trading_agent.json` - low-frequency dry-run configuration.
- `configs/etf_paper_trading_agent_execute.json` - low-frequency paper-execute configuration.
- `scripts/run_t0_intraday_agent.py` - T+0 intraday agent.
- `scripts/run_etf_paper_trading_agent.py` - low-frequency ETF paper agent.
- `scripts/replay_t0_decisions.py` - offline T+0 decision replay; no API calls, no orders.
- `scripts/run_holdings_calibration_report.py` - local holdings calibration report.
- `scripts/run_trading_self_review.py` - daily trading self-review.

## Safety Invariants

Do not weaken these without explicit review:

- Submit/cancel paths require `--execute`, `mode=paper_execute`, `execution_enabled=true`, and passing risk checks.
- `paper_trading_only=true`, `live_ready=false`, and `formal_strategy_allowed=false` must remain true/false as currently defined.
- Runtime outputs under `outputs/` are local diagnostics and are not tracked.
- `HT_APIKEY` must be provided through environment variables only.

## Validation Commands

```powershell
py -3.13 -m py_compile scripts/run_t0_intraday_agent.py scripts/replay_t0_decisions.py scripts/run_etf_paper_trading_agent.py
py -3.13 scripts/replay_t0_decisions.py --label local_check
```

The replay command must not write to `outputs/t0_intraday_agent/t0_state.json`.
