# Same-support fundamental Top10 training on Sina daily data

Research-only, retrospective diagnostic. No online inference, publication, order,
dashboard, frozen-model, forward-ledger or trading-configuration changes.

## Preregistered question

On a new, same-vendor raw/adjustment price basis, do conservative disclosed
fundamentals and four fixed interactions improve executable Top10 outcomes over
the SAME eligible fundamental baseline? This is not another search of the
456 price/volume factors and not a retraining of the old twelve-factor book.
Those formula discrepancies remain unresolved. This experiment cannot repair,
recertify or overwrite the old book by changing its name.

The old four interactions have already failed a historical incremental test.
Here the bounded reason for testing again is the corrected input basis and
explicit common support, not new hyperparameter searching. A failure is final
for this preregistration; no changing weights/windows to obtain a desired result.

## Inputs and excluded coverage

- Terminal collection `recovery_20260911_sina_v1`: 2019-01-02 through 2026-09-08.
  Collection ended with 5,791 / 5,798 symbols and 7 gaps. It is NOT certified
  training data merely because it finished.
- Require the full independent input audit first. Check saved file hashes,
  same-vendor raw/adjusted identities, shares/CNY units, OHLC and true aggregate
  VWAP arithmetic. No proxy VWAP and no mixing BaoStock prices with Sina prices.
- Use the collector's frozen archived PIT membership master. Retain historically
  available delisted names; do not require the latest price to exist.
- Join **only** same-date vendor ST/trade status from existing BaoStock files.
  No nearest-date or forward-filled status, and no network calls to that provider.
  BJ/current-only names lack verified historical coverage and are excluded,
  counted and reported. Do not describe this as all-A-share coverage.
- Use 500 prior observed price sessions, lagged 60-session liquidity/missingness
  gates and a minimum 300-name common cross-section, unchanged from the prior
  execution study. Same-day outcomes cannot alter eligible membership.
- Current vendor vintage, historical survivorship and revision completeness are
  still imperfect. This is an explicitly qualified historical subset, NOT a
  clean final OOS or daily recommendation.

## Fixed features and five arms

The existing four families and their 16 financial component definitions are
reused without editing the frozen module/config. Compute equal component means
within each family; each value is already a bounded `tanh` transform. Families:
earnings innovation (reported change, **not analyst-consensus surprise**), growth
acceleration, quality and cash-flow quality.

Disclosure availability is the first session strictly after max(noticeDate,
updateDate). reportDate only identifies fiscal periods. An available event is
carried at most 130 sessions. A newly disclosed missing component RESETs that
component to missing; it cannot silently inherit an older known value. All four
complete families and their 65-session-lagged copies are required for every arm.

Four fixed close-known contexts: 20-session momentum, negative 5-session return,
negative 20-session daily-return standard deviation, 5/20-session mean share
volume ratio minus one. No implicit price padding; 21 consecutive closes needed.
Cross-sectional ranks use identical complete support at that date, centered at
0.5. Interactions: earnings × volume, growth × momentum, quality × low volatility,
cash quality × reversal. No searched windows, orientations or interaction tree.

| Arm | Selection mechanism |
|---|---|
| equal_families | Equal mean of the four family ranks |
| fundamental_model | Trained payoff from four family ranks |
| context_model | Same learner, four families + four contexts |
| interaction_model | Same learner + four prespecified interactions |
| lagged_interaction_counter | Same current main effects, interactions use 65-session-old families |

Primary incremental comparison: interaction_model versus context_model. The
lagged interaction is a weak diagnostic counter, NOT a permutation significance
test (slow fundamentals remain autocorrelated). Report all arms, not just a winner.

## Training and execution

Evaluation begins 2024-01-02 and ends with all seven maximum-maturity sessions
available. Refit every 21 sessions: 252 training, gap 7, calibration 63, gap 7,
then next test block. Split whole dates, never randomly split overlapping rows.
All scaling, return clipping, learned coefficients and probability calibration
come only from the earlier train/calibration windows. Seeded maximum 300 rows per
training day, with equal aggregate weight per day; no outcome-based subsampling.

Reuse the existing logistic up head, conditional-gain/loss ridge heads and
Platt/return calibration. Add a calibrated <=-3% tail head only as a diagnostic,
not a selection gate. Fixed C=0.1, ridge alpha=10, 99.5% train-only clipping.
Rank model arms by calibrated expected gross payoff minus fixed 30bps cost.
Do not stretch probabilities or add a >50% floor. Equal-family picks receive the
same fundamental-model predictions for descriptive calibration only.

Signal at t close; entry t+1 open; exit t+2 or at the next sellable open within
five additional sessions. Raw opening/previous-close and dated status determine
opening availability, **never the session's later high/low/close/turnover**.
Use the documented historical board limit convention and fixed 0.5 percentage
point buffer. No queue reconstruction; even dated status availability at open
is an assumption. Corporate-action/unknown-factor legs are unresolved rather
than inferred across adjustment changes. Later ST exits also remain unresolved.

Select ten BEFORE checking future execution. Never replace unfilled names or
count unresolved exposure as zero-return cash. Report resolution coverage and
all-selected win bounds. Net mean needs ten resolved picks; paired comparisons
need both books complete on identical dates. Such conditional means can be
optimistic and are not evidence by themselves. All unresolved days block strict
  target acceptance. Cohort means are NOT account returns or funded portfolio PnL.
All requested evaluation sessions and all common-support days are also accounted
for: missing cross-sections or skipped fit blocks block target acceptance, even
if the remaining reported days look good.

## Reporting and stopping

- Every arm: fixed-count picks, probabilities, expected returns, observed gross
  and net returns, tail frequency, calibration metrics/buckets, annual breakdown.
- Same-support comparisons reuse the strict target evaluator: >50% realized
  win-rate lower bound and +1 percentage point net cohort return lift, positive
  net returns, no worse tail, complete execution; no daily guarantee implied.
- Hindsight best arm versus trailing-63-day best arm: compare net returns on
  identical joint-complete, matured days; at least 20 prior complete days needed.
  The difference measures selection exposure, not an implementable oracle.
- Five policy comparisons are counted globally AND for each of the four families;
  465 is only a global LOWER BOUND, not the true historical experiment total.
  DSR is null: complete global/family trial history is unavailable. There is no
  multiplicity certificate, no independent placebo distribution and no promotion.
- Save input hashes, code/config/environment, fold boundaries and coefficients.
  New run IDs only. No overwrites or automatic retraining after results.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/audit_sina_fundamental_readiness_v1.py --price-run recovery_20260911_sina_v1 --run-id audit_20260912_full5798
py -3.13 scripts/research_sina_fundamental_top10_v1.py --run-id run_20260912_frozen_fundamental_training_v1
py -3.13 -m unittest discover -s scripts -p test_sina_fundamental_top10_v1.py -v
py -3.13 scripts/test_replay_invariants.py
```

Outputs: `outputs/edge_research/sina_fundamental_top10_v1/<run_id>/`. A dataset or
fit failure is fail-closed and recorded; no data relaxation or next-model search
is authorized by a failure. Forward evaluation requires a separately frozen plan.
