# Handoff to the Codex line — measured constraints and revised priorities

Written by the Claude research line for the Codex line, which now owns all factor training.
Everything below is measured on the shared PIT-adjusted BaoStock panel, not argued. Full
detail: `docs/independent_net_factor_study_findings.md`, preregistration in
`docs/independent_net_factor_study_preregistration.md`.

Two owner constraints are taken as given and shape everything here:

- **History depth is not the lever.** A-share trading style shifts fast enough that older
  data misleads more than it informs.
- **Cost per trade is not negotiable.** ~30 bps round trip (commission 6 + stamp duty 5 +
  spread ≈ 19) is the market price for any factor.

## 1. What is already ruled out — do not re-run these

**Price-and-volume factors cannot clear the cost, at any horizon, under any weighting.**
The 456-factor vendored library (Kakushadze 101, GTJA 191, Qlib 158, academic) was scanned
end to end with point-in-time membership, ST/suspension exclusion, sealed-limit legs dropped
and 30 bps charged on realised turnover:

| horizon | gross bps/day | cost bps/day | net bps/day |
|---|--:|--:|--:|
| 1 | 3.84 | 1.84 | +2.00 |
| 5 | 2.43 | 1.22 | +1.20 |
| 10 | 1.71 | 1.02 | +0.69 |
| 20 | 0.33 | 0.80 | −0.47 |

Those net figures are the *hindsight* case (factors chosen after seeing the outcome window).
Chosen honestly on trailing information, every horizon is negative: −1.24, −2.36, −3.58,
−3.16 bps/day, all flagged `consistent_with_luck` by Deflated Sharpe at 454 trials.

The gross ceiling of roughly **4 bps/day** is the binding number. With cost fixed, no
weighting scheme and no additional price-volume factor can cross it. Lengthening the horizon
does lower cost (1.84 → 0.80) but loses gross faster (3.84 → 0.33), so that route is closed
too — this corrects an earlier suggestion of mine that longer holds were the way out.

Note on cost that is worth keeping in mind: the *rate* is fixed, but cost **per day** is
30 bps × turnover, and turnover is a strategy choice (0.65/day → 19.5 bps/day; 0.027/day →
0.8 bps/day). That dimension has already been swept; it does not rescue this library.

## 2. The largest measured effect in this project is the selection step, not any factor

Same panel, same cost model, same construction, same horizons. Only the factor-choice rule
differs:

- chosen **after** seeing the outcome window → **+2.00 bps/day** at h=1;
- chosen on **trailing net IR only** → **−1.24 bps/day** at h=1.

The ~3 bps/day gap is the selection artifact measured directly. It is larger than any edge
claimed anywhere in this project, which is precisely why an artifact of this kind reads as a
strategy.

**Requested practice:** any new selection or weighting method should report both numbers —
hindsight and trailing — before anything else. The gap is that method's overfitting exposure.

## 3. Style drift and ranking churn are the same phenomenon

Year-over-year churn of the trailing-net-IR top ten: **30% at h=1, rising to 50% at h=20**.
Read alongside the owner's point that A-share style keeps changing, this is not estimation
noise to be fixed with more data — the ranked object is itself moving.

Consequences that differ from earlier advice:

- **Selection by past performance should be dropped, not improved.** In a drifting market the
  trailing-chosen side of the gap above only gets worse. Equal weight across a mechanism
  family beats ranking, because ranking is the step that fails.
- **Validate mechanisms, not backtest performance.** Performance expires with the regime; a
  mechanism constraint (limited attention, forced liquidation, index-rebalance demand) does
  not. The existing Primary/Counter/Placebo design is right in intent — its problem is that
  every candidate underneath it is price-volume derived.
- **Weight training is the wrong next step.** If *which* factors to hold cannot be learned
  from trailing data, *how much* of each cannot be either: weights are the same estimation
  problem at finer grain with less information per parameter. For the frozen twelve-factor
  book specifically, the source run reports 0/9 validation and 0/9 shadow checks passed, and
  an independent scan finds 0/12 net-positive with 8/12 negative **gross**.

## 4. What is still open, in priority order

**(1) Fundamentals — the only untouched orthogonal information, already on disk.**
`data/market/ashare_research/fundamentals_pit/`: 5,782 files, median 47 quarters per name,
with `reportDate` and `noticeDate` for correct point-in-time alignment. Fit with both owner
constraints: fundamental signals decay on a quarterly clock, so they are naturally
low-turnover against a fixed cost, and mechanisms like post-announcement drift are far more
stable across style regimes than momentum/reversal, whose sign flips between regimes.

Suggested build: align on `noticeDate` (never `reportDate`), keep families separate —
earnings surprise, growth acceleration, quality, cash-flow quality — and **count trials per
family** rather than pooling into one search space, so the multiple-testing discount stays
honest.

**(2) Mechanism-family equal weight rather than top-N selection**, per section 3.

**(3) Freeze a forward record now.** No unseen historical holdout remains for either line —
the twelve-factor config states `validationAndShadowAlreadyViewed: true`, and this line has
seen 2025+ for the same library. Drift makes historical evidence decay further. Freezing
today's best version, even equal-weighted, and logging daily predictions makes roughly
2026-11 the first date with clean evidence. This is the only action that currently generates
uncontaminated data, so its value increases the earlier it starts.

## 5. Standing status

Zero validated factors exist in either research line. Nothing is connected to trading, and
nothing here authorises connection. Any positive result from this point needs post-2026-08-07
observation before it means anything.
