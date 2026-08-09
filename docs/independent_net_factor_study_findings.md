# Independent net-of-cost factor study — findings

Run against `docs/independent_net_factor_study_preregistration.md`. Rule fixed before the run:
a horizon is viable only with positive mean net excess, IR ≥ 0.5, drawdown ≥ −20%, and
survival of the Deflated Sharpe check at the full scanned trial count.

**Viable horizons: none.**

| horizon | OOS net (bps/day) | IR | max drawdown | win rate | DSR |
|---|--:|--:|--:|--:|---|
| 1 | −1.24 | −0.32 | −41.1% | 0.504 | consistent_with_luck |
| 5 | −2.36 | −1.33 | −37.8% | 0.457 | consistent_with_luck |
| 10 | −3.58 | −3.15 | −43.6% | 0.400 | consistent_with_luck |
| 20 | −3.16 | −4.14 | −36.3% | 0.382 | consistent_with_luck |

454 factors scored per horizon; top ten by trailing net IR held equally weighted for the
following year, forward segments concatenated.

## 1. Lengthening the horizon does not rescue this library

I had argued that a longer hold would amortise the 30 bps round trip and turn these factors
positive. Decomposed on a fixed factor probe, that mechanism is real but insufficient:

| horizon | gross bps/day | cost bps/day | net bps/day | turnover |
|---|--:|--:|--:|--:|
| 1 | 3.84 | 1.84 | +2.00 | 0.061 |
| 5 | 2.43 | 1.22 | +1.20 | 0.041 |
| 10 | 1.71 | 1.02 | +0.69 | 0.034 |
| 20 | 0.33 | 0.80 | −0.47 | 0.027 |

Cost per day does fall as predicted (1.84 → 0.80). Gross falls faster (3.84 → 0.33), because
the library is built from short-horizon technical factors whose signal is largely gone by
twenty days. Holding longer buys less cost and loses more signal. **My earlier recommendation
to lengthen the holding period is not supported by this test.**

## 2. The selection step is worth more than any edge measured in this project

The two tables above use the same panel, cost model, construction and horizons, and disagree
in sign. The only difference is how the factors were chosen:

- the probe's five factors were picked **after** seeing 2025+ results → **+2.00 bps/day** at
  h=1, still positive at h=10;
- the walk-forward chose by **trailing net IR only** → **−1.24 bps/day** at h=1, negative
  everywhere.

That ~3 bps/day gap is the selection artifact, measured directly rather than argued about. It
is larger than any edge claimed anywhere in this project, which is why an artifact of this
kind can comfortably masquerade as a strategy.

Why trailing selection fails is visible in the picks themselves: year-over-year churn of the
trailing-IR top ten runs 30% at h=1 and rises to 50% at h=20. Trailing net IR is not a
persistent ranking on this library, so selecting on it is close to redrawing each year.

## 3. What this settles about weight training

The proposal to retrain weights over the frozen twelve-factor book is answered empirically
rather than by argument. If **which** factors to hold cannot be learned from trailing data
(30–50% churn, negative forward result), then **how much** of each to hold cannot be either —
weights are a strictly finer-grained version of the same estimation problem, fitted on less
information per parameter. And because the validation and shadow windows for that book have
already been viewed, a weight fit evaluated against them would reproduce the +3 bps/day
hindsight gap as an apparent result.

## 4. What would actually change the answer

Not more factors, and not more optimisation over these factors. Only:

- **a different cost structure** — at 30 bps round trip the h=1 gross of 3.84 bps/day is
  already the ceiling; a commission-plus-stamp-plus-spread total near 10 bps would change the
  arithmetic, and that is a broker and execution question, not a research one;
- **a different information set** — fundamentals, flows, announcements, index membership
  changes. This library is price and volume only, and price-and-volume predictability on this
  panel is bounded at a few bps per day gross;
- **forward data** — no unseen historical holdout remains for either research line. Anything
  claimed from here needs post-2026-08-07 observation to mean anything.

## Standing status

Research-only throughout. No orders, no configuration change, no promotion. Zero validated
factors exist in either line.
