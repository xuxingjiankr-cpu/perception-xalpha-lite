# Opening-auction raw-rank abstention V3

This study corrects one specific V2 failure without changing any trading component.
V2 ranked calibrated outputs. Isotonic calibration is monotone but piecewise constant,
so a large set of stocks can receive identical calibrated values and therefore an
arbitrary Top10 order.

V3 separates two jobs:

1. **Ordering:** cross-sectional ranks of the three raw, purged walk-forward model
   heads retain relative discrimination. The V2 weights remain unchanged at
   55% probability-up, 30% expected-return and 15% tail-safety.
2. **Confidence and reporting:** train-only calibrated values determine whether a
   stock is eligible, but never determine its order. A stock must have calibrated
   probability-up of at least 52%, non-negative expected gross return and severe-loss
   probability no greater than 8%.

The policy may select fewer than ten names or abstain for an entire day. This is a
necessary part of the hypothesis: filling ten slots when evidence is flat mechanically
adds low-confidence observations.

Evaluation uses the same historical panel and 504-day training / 10-day outer purge /
63-day refit structure as V2. Every candidate cohort is compared with a prior-close
control that selects exactly the same number of names on the same dates and support.
This matched-coverage control exposes any apparent win-rate improvement caused only by
trading less often.

The window has already been viewed. Even if every historical gate passes, the only
permitted decision is a separately frozen fresh-forward study. The script does not
read or write trading configuration, create orders, call a broker, change positions,
or alter any execution or risk gate.
