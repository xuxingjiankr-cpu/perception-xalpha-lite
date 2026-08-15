# Hierarchical Opening-Auction Selector V7

## Status

Research-only and shadow-only. This trial cannot place orders or modify any
production decision, position, risk gate, overlay, or execution lock.

## Rationale

V6 showed that the train-only opening-auction breadth gate identified stronger
absolute market days, but replacing the established stock ranking with the V5
classifier reduced win rate relative to the same-day control. V7 therefore
assigns one fixed responsibility to each already-built component:

- V6 decides whether the day belongs to a reliable opportunity cohort;
- the existing prior-close score ranks the cross-section;
- the V5 calibrated tail head excludes only names whose severe-loss
  probability exceeds the already-frozen 8% ceiling.

No model, threshold, weight, feature, or window is newly fitted for V7. This is
one component ablation, not a parameter search. The candidate is compared with
the same prior-close ranking without the tail exclusion on identical dates,
support, and daily selection count.

## Interpretation

This hypothesis was chosen after observing V5 and V6, so its historical result
has material researcher-degree-of-freedom bias. Failure rejects it. A pass can
only justify preregistration and collection on future unseen days; it cannot
authorize trading.

