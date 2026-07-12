# ETF Trade-Success Shadow Research

Status: `research_only / shadow_only / no automatic promotion`

## Purpose

This package tests five ways to improve the success rate of the existing paper ETF
agent without changing its BUY/SELL gates, sizing, order path, risk gates, overlays,
or triple execution lock:

1. full-capacity replacement of a weak sellable holding by a stronger candidate;
2. prior-day-only weak-ETF cooldown memory;
3. candidate-rank monotonicity;
4. point-in-time full-market regime conditioning;
5. 5/10/20-minute weakness after confirmed BUY fills.

The frozen configuration is
`configs/research/trade_success_shadow_v1.json`. The daily runner is
`scripts/research_trade_success_shadow.py`. Artifacts are sealed by input hash under
`outputs/edge_research/trade_success_shadow/<run_id>/`.

## Why the earlier coarse aggregates cannot drive trading

The first 14 forward days contain 3,269 score rows, but the raw ledger needs cleaning:

- all nine score-ledger BUY rows have `was_executed=false` and no confirmed fill;
- 361 snapshot/code keys are recorded as both HOLD and BUY_CANDIDATE;
- all 2,445 candidate eligibility values are null in the historical score ledger;
- there are 235 dynamic `config_sha256` values because the observation universe is
  replaced at run time;
- the ledger's old `market_regime` value is always neutral;
- repeated intraday scans share the same later outcome and are not independent trades.

Therefore the new package uses the complete `t0_agent_runs.jsonl` snapshot for rank,
holdings, sellability, sell score, and full-market context. Confirmed BUYs come from
`paper_order_lifecycle.jsonl`. Decision-score rows are retained only for contamination
audit. A new `policy_sha256` metadata field excludes the dynamic universe and output
plumbing, so future score records can identify a stable policy cohort.

## Causal event construction

- The exchange-local decision timestamp defines the event time. Shanghai
  `source_quote_time` is audited and must not be later than that decision minute.
- Duplicate source minutes are deduplicated.
- Point-in-time snapshots are thinned to one every 20 minutes.
- A 5/10/20-minute target quote is accepted only within six minutes of its target.
- Candidate entry is modeled at the next ask and exit at the target bid.
- Future returns are labels only; they cannot select candidates, holdings, ranks, or
  cooldown state.
- Walk-forward groups the complete cross-section by trade date, starts only after 20
  training days, and uses one full trading day of purge plus one day of embargo. All
  promotion-readiness metrics use only the resulting test dates; full-sample numbers
  remain descriptive.

## Frozen hypotheses and evidence gates

### Capacity replacement

The candidate must rank in the top three, pass recorded execution/safety diagnostics,
and have a recorded `pre_capacity_entry_eligible=true`. The incumbent is the sellable
holding with the highest point-in-time sell score and must have sell score at least 65.
The comparison is:

`sell incumbent at bid -> buy candidate at ask -> 10m candidate bid`

versus continuing to hold the incumbent, after the fixed incremental cost. Discussion
requires at least 30 independent days, 50 events, ten candidate ETFs, a win rate of at
least 55%, positive mean, and a positive day-cluster 95% lower bound. This policy adds
two order legs, so it cannot improve mechanically by reducing trade count.

The paper agent now records the top non-held candidate's pre-capacity gate as metadata
when slots are full. The field is record-only, wrapped in exception containment, and is
not read by the decision or order path.

### ETF cooldown

The shadow state uses only the prior five completed ETF-days. It requires at least four
loss days and mean 10-minute net return at or below -0.30%. The shadow cooldown lasts one
trading day. Evaluation preserves coverage: a flagged top candidate is replaced by the
next non-flagged rank in the same snapshot. It is not compared with holding cash.

Discussion requires at least 20 trading days, 50 equal-coverage replacement events, ten
ETFs, and a positive day-cluster lower confidence bound. Candidate evidence can never
promote a production blacklist; confirmed fills and separate future OOS are required.

### Candidate rank

The primary metric is 10-minute net return of rank 1 minus the mean of ranks 2 and 3 in
the same snapshot. Secondary metrics are within-snapshot rank correlation and frozen
rank buckets. Discussion requires at least 20 days, 200 paired snapshots, at least
10 bps economic separation, a positive clustered interval, and the correct sign in at
least three of four weekly blocks.

### Regime conditioning

The old all-neutral score-ledger regime is not used. The package reads the point-in-time
`full_market_entry_guard` detail and reports risk-on, risk-off, neutral/selective, and
unavailable groups. No group can become a gate before its fixed independent-day and
symbol-day minimums are met in future data.

### Post-entry weakness

Only lifecycle rows with confirmed fill evidence are primary. The main horizon is ten
minutes; five and twenty minutes are secondary. Unknown exact fill times are disclosed
and may use submission time only for markout research. At least 30 confirmed BUYs across
20 days are required before discussing a conditional exit. Candidate rows cannot satisfy
this requirement.

## Daily operation

`scripts/run_research_suite.ps1` runs the shadow study immediately after the daily score
enrichment. It writes only to its independent research output directory. Re-running the
same `run_id` and input hash is deterministic and atomic; a different input fingerprint
cannot overwrite a sealed run.

## Current interpretation

The first cleaned run remains diagnostic-only. Rank-1 has no stable 10-minute advantage,
candidate 10-minute returns are negative after cost, cooldown has no eligible equal-
coverage evidence, and historical capacity snapshots lack the newly recorded
pre-capacity gate. Confirmed BUY markouts remain too few for an entry-failure rule.

Accordingly, this package makes no trading recommendation and changes no paper-trading
behavior. The already enabled risk-aware 3% take-profit remains separate from this work.
