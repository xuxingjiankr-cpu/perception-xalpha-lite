# Daily Top10 discrimination protocol V2

> **Preregistered research-only / shadow-only protocol. No order path and no automatic
> promotion.**

## Direct objective

The system succeeds only if the ten stocks selected at each close subsequently exhibit
both a higher probability of rising and a higher executable net return than stocks not
selected. Factor IC, a positive full-universe regression coefficient, or a profitable
post-hoc component is not sufficient.

The frozen V1 equal-component/equal-family PIT score is unchanged. V2 changes the
evaluation target from a top decile to exactly ten names; it does not fit a new weight,
select a historical winner or reopen price-volume factor search.

## Executable outcome

- Score after market close on session `t`, using information then available.
- Attempt entry at the next buyable open (`t+1`).
- Exit at the next sellable open (`t+2`), respecting A-share T+1.
- Charge 30 bps round trip.
- Choose ten names before observing executability. A locked or suspended selection is
  unresolved/unfilled and is never replaced by rank 11.

## Comparisons

Each date produces paired Top10-minus-control observations against:

1. every other eligible stock; and
2. a control matched by board and causal trailing-liquidity decile.

The report includes gross-up probability, net-positive probability, loss probability,
mean and median net return, paired return lift, daily outperformance rate, outcome
coverage and day-clustered HAC statistics. Stock-day counts are descriptive only; the
inference unit is the trading day.

## Acceptance

All preregistered gates in
`configs/research/fundamental_top10_discrimination_v2.json` must pass on at least 60
independent post-2026-08-07 forward sessions. In particular, both probability lifts and
both return lifts must be positive, Top10 mean and median net return must be positive,
Top10 loss probability must be lower, and the relevant HAC t-statistics must reach 2.0.

Historical observations end at 2026-08-06 for the frozen diagnostic calibration prior.
They can reject the score but cannot validate or promote it. Forward observations cannot
refit V2. A changed score, threshold, cost or calibration requires a new version and a
new preregistration.
