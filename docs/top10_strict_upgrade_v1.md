# Top10 upgrade: strict inputs before another model search

Status: research-only / shadow-only. No production, dashboard, broker, risk-gate,
weight, probability-artifact, forward-ledger or scheduled-task changes.

## Scope and findings

The latest historical payoff-decomposition comparison did not improve the
Top10. In its reused `shadow` window the old guarded16 book had a 47.23%
resolved-pick up rate and -0.0781% gross per pick; payoff decomposition had
45.92% and -0.1289%. These are historical cohort statistics, not today's
forecasts. Larger gains came with larger losses, not better direction.

Read `handoff_to_codex_factor_research.md` before extending this work. Do not
restart the exhausted 456-factor price/volume sweep or retrain weights until
these prerequisites are addressed.

Confirmed implementation issues:

1. The BaoStock collector labels adjusted OHLC4 as a VWAP proxy. The legacy
   `research_perception_xalpha_horizon_precision_v3.build_factor_inputs()`
   subsequently overwrites VWAP with raw amount / raw volume. That value is
   inconsistent with backward-adjusted OHLC and affects price-relative formulas.
2. Vendored `alpha101/alpha_094.py` describes `lhs ^ rhs` in its formula metadata,
   but executes `(lhs * rhs) * -1`. This is a metadata/implementation discrepancy,
   not yet a source-paper reproduction audit. Neither version is silently chosen.
3. Vendored `alpha101/alpha_029.py` uses `close.pct_change()` without disabling
   implicit padding. The new input builder disables padding for its own returns
   but does not claim to fix internal missing-data behavior of old formulas.

The old 12/16-factor definitions, weights and reports remain untouched.
An input-basis correction is a **new research version** and needs an identical-
support baseline. Old results cannot be relabelled as corrected results.

## Implemented

- `research_top10_strict_inputs_v1.py`: opt-in price-pair validation, explicit
  numeric input allowlist, real vendored factor computation, complete-support
  ranks, and streaming per-symbol price audit with input hashes.
- `research_top10_upgrade_readiness_v1.py`: isolated audit and evaluation of the
  user's numerical targets against an existing historical picks CSV. Never fits
  a model or publishes a new Top10; blocked prerequisites produce exit code 2.
- Frozen experiment/acceptance contract in
  `configs/research/top10_strict_upgrade_v1.json`.
- T168 exercises ten tests, including actual GTJA131 and all current twelve
  module entry points. Synthetic smoke outputs are never alpha evidence.

### Price contract

An unadjusted companion file is named `SH_600000.jsonl`, for example, under
`data/market/ashare_research/baostock_pit_raw_companion_v1/`. The directory is
an input location only: this release does not populate it or start a collector.
Raw rows need the same normalized fields as the existing adjusted rows, but
`adjustflag="3"` and `adjustment="none_raw_baostock"`. Source must be
`baostock_query_history_k_data_plus`; raw OHLC, shares, CNY amount, and
date-specific trade/ST status must be genuine vendor values, not relabelled TDX.

For each date and symbol independently:

1. Verify source, units, active status, exact identities, finite positive values,
   valid OHLC ordering, duplicate-free input and stable file content.
2. Require raw and adjusted amount/volume to match (relative tolerance 1e-8),
   and status to match. Carried-forward status is not accepted.
3. Require the four adjusted/raw OHLC ratios to agree (relative tolerance 1e-5).
4. Require raw amount / shares to lie inside raw low/high (tolerance 1e-5).
5. Reconstruct `adjusted_vwap = raw_amount / raw_shares * adjusted_close / raw_close`.

This is a checked multiplicative reconstruction, not a vendor-issued adjustment
factor or an executable fill. A missing pair stays NaN. No OHLC4 substitution,
close fallback, nearest-date pairing, cross-provider scale inference, or missing
factor median completion is allowed. A complete exchange-session grid is needed.
PIT listing/seasoning/eligibility must still be supplied by the opt-in PIT loader.
The new adapter also requires 252 consecutive verified OHLC/flow/VWAP input
sessions, frozen before evaluation. This prevents partial warmups and hidden
padding across gaps from manufacturing complete observations for this book;
it is an input exclusion, NOT evidence of better returns. A later baseline must
use exactly the same exclusion. All selected factors, even zero-weight factors, must have finite values; all ranks
use the same complete cross-section. Formula internals remain unaudited and block
training certification. Historical vendor revisions and cash dividends are not
fixed by this arithmetic. Coverage currently concerns SH/SZ, not BJ.

## What '>50% and +1%' means

The unanswered clarification is provisionally interpreted as **one percentage
point additional mean Top10 return per identical holding period versus baseline**,
not a 1% relative increase, not a guaranteed +1% daily account return.

- Report each model's individual probabilities unchanged. If any selected name
  has predicted up probability <=50%, that prediction target fails. Never lift
  a probability by normalization, stretching or an arbitrary floor.
- Separately measure real gross-up frequency. The lower one-sided 95% bound
  must exceed 50%, with at least 20 independent signal days. Ten stocks from
  one day are not ten independent days. Zero gross return is not a win.
- Candidate-minus-baseline mean **net** cohort return must be >=0.01 and its
  lower bound >0; candidate net return must be positive; the <=-3% gross tail
  frequency must not worsen. Both books use fixed 30bp round-trip cost.
- Select exactly ten **before** checking next-session opening execution. No
  post-outcome replacement, smaller favorable subset, or deleting bad days.
  Signal at t close, buy t+1 open, sell no earlier than t+2 open (A-share T+1),
  with the existing maximum five-session exit delay and seven-session purge.
- Price-limit/queue and vendor-vintage assumptions must remain explicit. No
  daily high/low may determine an opening fill.
- Return comparisons use paired days where all ten outcomes resolve in each
  book. Unresolved selections remain in the denominator and block acceptance;
  they are never zero-return cash. Up-rate bounds assume unresolved names are
  all nonwinners / all winners. Report complete-day coverage prominently.
- Uncertainty uses a frozen seven-day circular moving-block bootstrap (2,000
  replicates, seed 20260908). Calibration metrics are descriptive pooled-pick
  Brier/LogLoss/AUC/ECE and ten fixed probability buckets, not proof of daily wins.
- Historical windows have been reused. Even a numerical pass CANNOT certify
  alpha or authorize trading. Multiplicity and fresh-forward evidence remain
  separate requirements, with global AND mechanism-family trial accounting.

## Next model, after prerequisites — not implemented in this release

1. Qualify first-seen, publication and availability timestamps with source URLs,
   document hashes, revisions and security mappings. Quarterly ratios and today's
   retrospective news summaries are not substitutes for a historical event feed.
   `reportDate` is never event availability. For historical archives, firstSeenAt
   means verifiable original availability, not a fabricated historical timestamp.
2. Treat novelty/first-vs-followup/rumor-vs-confirmed as event metadata hypotheses.
   Missing news coverage cannot become a 'no event' signal. Announced-minus-
   expected earnings requires a real **pre-event** consensus vintage; otherwise
   skip that feature, not call sentiment or earnings growth 'surprise'.
3. On identical eligible samples compare repaired frozen16, frozen16 plus verified
   event metadata, and a lagged-event counter. Training/calibration 252/63 sessions,
   seven-session purge and 21-session refits stay fixed. Specify the actual model,
   loss and feature transforms in a separate preregistration before running it.
4. Report hindsight vs trailing-only net results and their gap. No current-window
   target-driven factor/weight sweep. Foundation models and component-decomposition
   architectures remain deferred comparators, not installed or enabled here.

This follows the prior literature review's information-first hypothesis. It does
not claim to reproduce any paper or inherit its reported returns. References:
[news-event study](https://arxiv.org/html/2608.14014v1),
[ACT](https://arxiv.org/html/2604.20204v1),
[FactorMiner](https://arxiv.org/abs/2602.14670).

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_top10_upgrade_readiness_v1.py --run-id run_20260908_strict_inputs_v1 --diagnostic-picks outputs/edge_research/top10_payoff_decomposition_v1/run_20260908_payoff_v1/daily_picks.csv
py -3.13 -m unittest discover -s scripts -p test_top10_strict_upgrade_v1.py -v
py -3.13 scripts/test_replay_invariants.py
```

Output: `outputs/edge_research/top10_strict_upgrade_v1/<run_id>/` containing
manifest, price audit, historical target diagnostic, result and short report.
Existing run IDs fail closed. No old artifact is overwritten. The manifest
records commit/dirty status, content hashes, environment and comparison arguments.

Current external issue: the single BaoStock login probe on 2026-09-08 returned
provider code `10001011` / `黑名单用户，请与管理员联系`. This is a provider failure
even though the probe process itself exited zero. No repeated/bypass attempts
are authorized by this release. Obtain restored legitimate access or supply
verified original source data before claiming the price history is repaired.
