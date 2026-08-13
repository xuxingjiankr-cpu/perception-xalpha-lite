# Sixteen-factor walk-forward weight protocol V1

This research asks whether the twelve price/volume factors and four PIT
fundamental interactions should receive non-equal ranking weights. It does not
assume that a higher historical return alone identifies a correct weight vector.

## Frozen method

- Inputs: the same 16 oriented cross-sectional ranks used by the existing
  sixteen-factor challenger.
- Outcome: executable next-open to following sellable-open gross return.
- Objective: daily-equal pairwise Top10 ranking utility, with explicit penalties
  for non-positive and severe-loss outcomes.
- Estimation: 504 trailing sessions, refitted every 63 sessions, with a ten-session
  purge between fit and test.
- Bounds: every factor remains between 1% and 15%; the four-interaction block
  remains between 15% and 40%.
- Regularization: weights shrink toward the frozen prior rather than an
  unconstrained historical optimum.
- Stability: five 21-session block subsamples report weight ranges.

The same dates are evaluated using the current ranking, honest trailing weights,
and an invalid same-period hindsight fit. Hindsight is reported only to disclose
selection exposure and can never be published.

## Acceptance

A learned vector must improve mean gross return and gross up-rate, must not
worsen severe-loss frequency, and must improve gross return in a majority of
walk-forward blocks. Failure of any gate retains the existing frozen ranking.

All historical windows have already been viewed. A historical pass could only
create a separately preregistered forward challenger. This module never updates
the dashboard, trading configuration, risk gates, positions, orders or execution
locks.
