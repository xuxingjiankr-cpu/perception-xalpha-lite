# Independent net-of-cost A-share factor study — preregistration

Registered **before** the run, by the Claude line, deliberately separate from the Codex
perception/XAlpha line. Shared inputs: the PIT-adjusted BaoStock panel only. Nothing else is
shared — separate scripts, separate output namespace (`outputs/independent_research/`),
separate decision rule. This line never reads or writes Codex state, configs or registries.

## Why this exists

The Codex line has produced, across v1–v10, zero factors that pass its own validation gates,
and the twelve-factor book now proposed for weight training carries a source verdict of
0/9 validation and 0/9 shadow checks passed. My own independent scan of the same 456-factor
library at a one-day hold reproduced that: 0 of those 12 are net-positive out of sample, and
8 of 12 are negative *gross*.

Both lines therefore share one unexamined assumption: that the failure is a factor-quality
problem. The evidence points elsewhere — outcomes ordered by turnover, not by signal
strength — so this study tests the cost/horizon structure directly instead of searching for
a better factor at a fixed horizon.

## Preregistered question

**At what holding horizon, if any, does a factor selected purely on trailing information
earn a positive net-of-cost excess return out of sample on A-shares?**

Not "which factor is best" — that question is what produces selection artifacts. The object
of study is the *horizon*, with factor choice made mechanically inside each window.

## Method (fixed before running)

- **Panel**: PIT-adjusted BaoStock, backward-adjusted prices, point-in-time ST and
  trade-status, delisted names retained.
- **Membership**: per date, trailing-60-session median amount (shifted) ≥ ¥30M, ≥120 prior
  observations, bar actually traded, not ST, status normal.
- **Tradability**: sealed bars (high == low) are locked limits — a sealed-up entry cannot be
  bought, a sealed-down exit cannot be sold. Those legs are dropped, not priced.
- **Horizons tested**: 1, 5, 10, 20 trading days. Entry at open[t+1], exit at
  open[t+1+h]. Overlapping tranches: 1/h of the book turns over each day.
- **Cost**: 30 bps round trip on realised turnover (commission 6 + stamp duty 5 + spread),
  charged identically at every horizon. The horizon changes turnover, not the rate.
- **Factor library**: the vendored 456 (Kakushadze 101, GTJA 191, Qlib 158, academic).
- **Selection**: strictly walk-forward. At each annual rebalance, rank factors by their
  **trailing net-of-cost IR only** and hold the top K = 10 equally weighted for the next
  year. No weight optimisation — equal weights, deliberately, so the result cannot be a
  weight-fitting artifact.
- **Evaluation**: the concatenated out-of-sample year segments form one continuous track
  record. Selection never sees the segment it is evaluated on.

## Decision rule (fixed before running)

A horizon is declared **viable** only if all four hold on the concatenated walk-forward
out-of-sample track:

1. mean net excess > 0;
2. annualised net IR ≥ 0.5;
3. maximum drawdown ≥ −20%;
4. it survives the Deflated Sharpe check at n_trials = number of factors scanned.

If no horizon passes, the reported conclusion is that the cost structure — not factor
quality — is binding, and no amount of further factor search or weight training at these
horizons is justified.

## What this study cannot settle

- Validation and shadow windows have already been viewed by the Codex line, and I have
  already seen 2025+ for this factor library. **No unseen holdout remains in history.** The
  walk-forward design reduces but does not eliminate that: the library itself was assembled
  by people who saw this market. The only clean adjudicator left is data after 2026-08-07,
  and any positive finding here is a hypothesis for that period, not a conclusion.
- Survivorship is improved (delisted names retained) but historical index membership and
  full ST history remain incomplete.
- No fill model: participation limits, queue position and market impact are not simulated.

## Standing constraints

Research-only. This line emits no orders, touches no trading configuration, and cannot
promote anything. A positive result is a hypothesis for forward observation, never a
trade signal.
