# Next-session factor-zoo V2 preregistration

## Scope

This is an offline, research-only attempt to improve the shortest executable A-share
forecast: form a signal after day `t` closes, hypothetically buy at the first buyable
open on `t+1`, and sell at the first sellable open on `t+2`. It cannot call a broker,
create orders, alter positions, write an overlay, or feed a production decision gate.

The two objectives are deliberately separate:

1. raise the executable gross return; and
2. lower the probability of a non-positive or severe-loss outcome.

A factor that improves only rank correlation does not qualify.

## Frozen discovery protocol

The declared library contains 456 implementations from Alpha101, GTJA191, Qlib158 and
an academic set. Discovery uses a deterministic 1,200-name screening cross-section and
only dates assigned to the existing training period.

For each usable factor:

1. Determine its direction from the first half of training only.
2. Apply that frozen direction to the remaining training observations.
3. Require an orientation-period absolute Newey-West t-statistic of at least 3.0.
4. Require a confirmation t-statistic of at least 1.65 and the same direction in at
   least three of four chronological confirmation folds.
5. Apply Benjamini-Hochberg FDR at `q <= 0.10` across the declared search.
6. Require both positive gross top-decile excess return and a lower non-positive rate
   than the contemporaneous eligible universe. Severe-loss improvement may be zero but
   not negative.
7. Reject turnover above 0.90 per day.
8. Greedily remove candidates whose daily IC correlation exceeds 0.75 with a stronger
   retained candidate.
9. Retain at most 12 factors.

No validation or shadow outcome may select a factor, direction, weight or threshold.
The search is counted as a new research trial even if no factor survives.

## Multi-target model

Surviving factor ranks are auxiliary inputs alongside the frozen V1 price/volume and
fundamental features. The candidate universe is the union score of the legacy burst
score (35%) and the train-selected factor score (65%), capped at 1,000 names per day.

Five separately calibrated probability heads estimate:

- executable return at least 5%;
- next-session board-normalised limit touch;
- gross return at or below zero;
- return at or below the 30 bps round-trip cost; and
- executable return at or below -3%.

Separate regressors estimate expected return and its conditional tenth percentile.
Every head must pass a nested training-only reliability audit before any row can pass
the selection policy.

## Fail-closed selection

A selected row must simultaneously satisfy the frozen return, gross-loss, net-loss,
strong-gain, limit-touch, severe-loss and tenth-percentile thresholds. At most ten rows
may be returned, and zero is a valid result. Top3 and Top1 are strict prefixes of the
qualified Top10; diagnostic rankings cannot be relabelled as recommendations.

## Interpretation

The external validation and shadow windows have already appeared in earlier research.
They can reject V2 but cannot provide clean final evidence or authorize trading. Even a
numerically successful V2 would require a separately preregistered fresh-forward study.
