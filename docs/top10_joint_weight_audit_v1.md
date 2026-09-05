# Joint Top10 weight audit V1

Research-only. Six new weight policies, not a new factor search. No dashboard,
trading configuration, order, execution lock or frozen forward record is changed.
Historical results cannot validate, promote, or guarantee a profitable Top10.

## Preregistered question

Can factor redundancy control and a joint return / up-rate / tail objective improve
both Top10 average return and the fraction rising, relative to the current guarded
sixteen-factor score? Do any improvements survive removing volatility exposure?

The twelve existing oriented price-volume ranks plus the four existing PIT
fundamental-price interactions are reused unchanged. All sixteen must be present;
no missing-factor imputation or alternative factor selection is permitted. Therefore
this audit's common universe is stricter than the imputed dashboard's universe.

## Six policies and three controls

Controls: frozen 16-factor prior, equal 16, current guarded 16 (the primary control).
Policies: covariance decorrelation, joint balanced (40/40/20), win tilted (20/60/20),
return tilted (60/30/10), volatility-neutral balanced and volatility-neutral win.
The triples are **training-task weights**, not final factor weights. Final factor
weights are fitted on a positive simplex, each between 1% and 25%.

Training moments equally weight each session. Within a session, 60% of observation
mass goes to the prior's Top100 neighbourhood and 40% to the full universe.
Returns, up labels and negative severe-loss labels are centred and scaled inside
the training day. Returns are scaled by 2% and clipped at +/-3 before centring.
This is a constrained surrogate for Top10 utility, not an exact Top10 optimizer.
The 16x16 feature covariance is shrunk 25% towards its diagonal; a 0.1 quadratic
penalty anchors weights to the prior. SLSQP solves the bounded quadratic objective.
For volatility ablations, each day's feature ranks are residualized against its
past-20-session realized-volatility rank. No future return enters this step.

## Timing and comparison contract

- Rank at session t close before examining future entry/exit eligibility.
- Entry t+1 open if buyable; exit at first sellable open at/after t+2, up to five
  additional sessions. This respects A-share T+1, unlike new-buy same-day exits.
- Weight refit every 21 scored sessions from 252 trailing sessions. Require 200
  usable training days. Purge seven sessions (maximum label maturity). The last
  possible training exit is strictly before the first test decision.
- Fundamental availability is inherited: first market session strictly after
  max(noticeDate, updateDate), never reportDate.
- Exactly ten selections; unresolved or unbuyable names are not replaced by lower
  ranks using hindsight. Report resolved-only metrics AND missing/selection counts.
  The fixed-slot net proxy assumes zero for missing returns solely as a sensitivity
  diagnostic; this is not a liquidation value for unresolved positions.
- Keep the inherited audit / validation / shadow dates. Each period censors labels
  whose exits cross its end. Every historical period has already been viewed.
- Charge 30bp round-trip; report gross, net, gross/net win, <=-3% tail and daily
  excess over the same-support resolved universe. Cohort returns are not an account
  equity curve, annual return or capital-constrained backtest.
- Paired daily Newey-West/HAC (7 lags), monthly stability, and six-policy Holm
  correction of the joint return-and-win test. Require improvement of both, no worse
  tail, positive net mean, >=95% resolved coverage, and majority joint-positive months
  in BOTH validation and shadow to survive this reject-only screen.
- Fit an explicitly invalid hindsight ceiling on all evaluated matured days; score
  on the identical period dates/cost. Report hindsight-minus-trailing net exposure.
  Neither this ceiling nor observed historical winners may be published to trading.
- Six new policy trials; 454 earlier trials are only a historical lower bound.
  A complete prior trial-return ledger is unavailable: DSR/PBO are null, not falsely
  passed. Paired tests do not erase factor-selection or repeated-window bias.

## Reproducibility

Run `py -3.13 scripts/research_top10_joint_weight_audit_v1.py --run-id <unique-id>`.
The exclusive output directory contains frozen config, code/config hashes, parent
commit and dirty status, panel cache identity/source audits, daily baskets, historical
weight paths, six latest research weight vectors, and reports. Output collisions fail.
Existing content-addressed panel/rank caches are reused, not existing result files.

Tests: `py -3.13 scripts/test_top10_joint_weight_audit_v1.py` tests the actual `run()`
entry on a synthetic panel, frozen-history causality, no future-mask replacement,
simplex bounds, volatility residuals and training-day market neutrality. Existing
`scripts/test_replay_invariants.py` must remain green separately.

No new factor directions, online inference, parameter search beyond the six named
policies, automatic promotion, or probability stretching is authorized by this study.

## Method references and regression isolation

The covariance regularizer is inspired by [Ledoit and Wolf's shrinkage work](https://ledoit.net/honey.pdf).
This implementation uses a fixed 25% diagonal shrinkage for factor features; it is
**not** a fitted Ledoit-Wolf stock covariance estimator or a reproduction of the
paper's performance claims. The execution timing follows the non-revolving-stock
restriction in [SSE Trading Rules section 3.1.4](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml).

The first full regression run exposed a pre-existing T23 fixture conflict: a newer
profit-taking rule exited a fixture holding before its unrelated sector-entry
assertion. T23 now disables that rule **only in its in-memory test config**. No disk
trading config or production risk code changes. T165 invokes all six new tests,
including the synthetic full research entry, as part of replay invariants.

## Confirmed inherited data-contract limitations (not fixed by weight fitting)

Review during this run confirmed that `research_ashare_universe.py` retains symbols
using full-file median turnover, full-file suspension/missing fractions, and a
minimum total observation count before constructing historical panels. Thus PIT
announcement/status alignment does **not** make the entire research universe
selection point-in-time safe. This run compares all policies within the same
preselected pool; it does not repair that upstream selection bias and is NOT clean
OOS. The cache audit's `unbiasedHistoricalValidationEligible: true` must not be read
as independent proof that this bias is absent. A future point-in-time-universe
study needs a separately scoped data rebuild; changing weights cannot fix it.

The cache key includes source/config hashes and raw-tree size/mtime signatures;
its per-frame "checksum" is structural, not a hash of all numeric cell values.
It is a useful acceleration/provenance aid, not exhaustive corruption detection.
Fundamentals also lack complete historical restatement vintages. These limitations
apply even if a candidate clears every statistical test reported here.
