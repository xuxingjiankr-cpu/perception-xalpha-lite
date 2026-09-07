# Top10 payoff decomposition — preregistered research V1

Status: research-only; no orders, model promotion, dashboard publication, or frozen-forward changes.

## Question and prior evidence

The published sixteen-factor composite ranks stocks independently of the displayed
expected return. Its near-49% up estimates are not a high-win-rate trading model.
The September 6 six-policy weight audit failed to jointly improve return, wins
and tail risk. We do not repeat that search, change factor definitions, extend
holding periods, or stretch probabilities. Read `handoff_to_codex_factor_research.md`.

One new hypothesis: conditional winning/losing payoff magnitude can add ranking
information beyond a directly fitted mean-return head on the SAME sixteen features.
An up-probability ranking is an ablation, not a second search for the best weights.

## Data repair, without changing frozen consumers

`research_ashare_pit_panel_v2.py` is opt-in. The legacy loader/config/cache remains
untouched. Remove whole-file admission tests for median amount, total length,
suspension, missingness and bad future status. Instead: 500 prior valid observations,
60-session lagged liquidity window (minimum 48 valid observations), CNY30m median,
lagged missing fraction <=15%, suspension fraction <=20%, daily verified non-ST,
trading and listing/delisting membership. These values are frozen before execution.
No future-appended bad or illiquid rows may change past membership.

Use an exchange-calendar session grid, not weekday approximations or compressed
dates after dropping missing bars. Mask substituted TDX sources and carried-forward
ST/trading status. Hash every source file's numeric-content-containing bytes.
No cached `unbiased` flag is accepted as proof: current-file statement revisions,
incomplete historical membership vintages and source coverage remain limitations.

The existing price-factor definitions and raw-amount/volume VWAP construction are
retained, and their possible incompatibility with adjusted OHLC is disclosed. The
four fundamental families still use the first market date strictly AFTER the later
of notice/update dates. No report-date availability or synthetic fundamentals.

On 2026-09-08 the live-data diagnostic returned BaoStock code 10001011:
"黑名单用户，请与管理员联系". This is an upstream access restriction, not a parameter
problem. Do not bypass it or label proxy/carried-forward data freshly verified.
Historical research can proceed; new live-ranked output cannot be certified current.

## Frozen experiment

- Input range: 2019-01-02 to 2026-09-04, SH/SZ PIT-master names; no BJ claim.
- Same existing audit/validation/shadow dates; all were previously viewed.
- All sixteen factors complete; no median imputation. Same daily support and ten
  choices per policy before inspecting future fill/outcome availability.
- Controls: guarded16, frozen16, equal16; prediction-head ablations: direct return,
  up probability, conditional payoff. No best-of-grid selection.
- Rolling 252 training sessions, 63 calibration sessions, 7-session purge between
  training/calibration and calibration/test; refit every 21 calendar trading sessions.
- Deterministic maximum 300 observations/day for fitting, day-equal weights; all
  complete names evaluated. Minimum training/calibration usable days: 180/40.
- Train-only standardization; logistic C=0.1; Ridge alpha=10. No class rebalancing.
- Up = gross return >0; non-up includes flat outcomes. Fit winning magnitude and
  non-up loss magnitude separately; retain zero losses in the non-up regression.
- Training 99.5% absolute-return quantile caps target magnitude for numerical
  stability; actual evaluation returns are NEVER clipped. Return unit =0.02.
- Platt up calibration and affine Ridge return calibration (alpha=10) fit only the
  purged calibration window. Apply the same procedure to direct/decomposed heads.
- Raw decomposition: p(up)*E[gain|up] - (1-p(up))*E[loss|non-up]. Calibrated return
  minus fixed 30bp cost determines the payoff ranking; no risk-weight grid.

## Execution and inference discipline

Signals at t close; next session open entry; earliest sale t+2 open, max five-session
exit delay. Opening execution proxies use open/preclose, board/date limits,
membership and date-level status, NOT that session's later high/low/volume/close.
Use conservative 0.5pp clearance from the limit, with board/date/ST distinctions;
exact tick-rounded limit prices and auction queues are unavailable. This remains
an execution assumption, not verified fills. Do not require tomorrow's liquidity
eligibility to retroactively decide today's picks. No future replacement.

Distinguish unfilled entry, pending maturity, and entered-but-unresolved exit.
Do not turn unresolved exits into zero-return cash. Per-cohort results are not a
capital-constrained portfolio curve. Also report resolution coverage/attrition.

Report gross/net return, gross/net win rate, mean win/loss, profit/loss ratio,
worst 5% mean, >=3% loss rate, per-month and board breakdowns, full-support up
Brier/AUC and return MSE/rank correlation. Primary: payoff must improve net return
versus BOTH direct-return and guarded16, with no worse severe-loss rate, positive
net mean, >95% resolved coverage and improvement across a majority of months in
BOTH reused evaluation windows. HAC(7) paired tests; Holm across three head ranking
policies. No success permits promotion. DSR/PBO stay null without a complete trial
ledger/appropriate experiment matrix; historical trial lower bound 460 + 3 policies.

Report a deliberately invalid all-evaluation-data fitted/calibrated hindsight
comparison on the same dates/support/costs. Its gap versus rolling is a leakage
sensitivity diagnostic, not a causal decomposition or guaranteed mathematical bound.

## Acceptance and artifacts

Independent directory: `outputs/edge_research/top10_payoff_decomposition_v1/<run_id>/`.
Manifest binds config, dependencies, commit, input-content hashes, calendar and
environment; preserve all original artifacts. Daily choices and predictions are
offline research evidence, not appended to any clean-forward ledger.
Tests must cover actual run entry, input-tail invariance, strict label maturity,
train/calibration/test isolation, next-open execution and no future replacement.
Run the full existing replay invariant suite before committing.

No further parameter adjustment from this experiment's outcome is authorised.
