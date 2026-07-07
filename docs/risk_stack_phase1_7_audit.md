# Risk Stack Phase 1.7 audit

## Objective

Phase 1.7 evaluates whether a causal risk stack can provide useful context for
exit tightening and position de-risking:

```text
BOCPD -> Koopman/DMD -> HMM -> Hawkes -> exit / position risk context
```

This is not a BUY/SELL model. The output is a research-only risk context, not a
trade gate.

## Literature grounding

- BOCPD: Adams and MacKay (2007), online changepoint filtering.
- DMD/Koopman: Schmid (2010), DMD as a data-driven dynamical diagnostic.
- HMM/regime: Hamilton (1989), Markov regime switching.
- Hawkes: Hawkes (1971) and Bacry, Mastromatteo and Muzy (2015), self-exciting
  event intensity in finance.
- Exit/position layer: triple-barrier style labels and risk-conditioned exits.

The literature supports the modeling roles, but does not validate an A-share ETF
T0 edge. Local replay and forward validation remain mandatory.

## Phase 1.7 implementation boundary

The first implementation is an audit and minimum viable historical diagnostic:

- BOCPD is a fixed-hazard Gaussian online filter on completed market returns.
  Its alert feature is the posterior probability of a short run length, not the
  raw constant-hazard `run_length=0` probability.
- DMD features are reused from Phase 1.6 and are not imputed.
- HMM features are reused from Phase 1.6 and are not refit.
- Hawkes is a fixed discrete-time exponential self-exciting intensity over
  observable risk events.
- Labels remain offline future labels and are never used in features.

No model output may alter paper trading, order generation, exits, position
sizing, risk gates, overlays, or forward frozen models.

## Validation questions

The audit asks:

1. Is each feature strictly past-only?
2. Is there enough event density for Hawkes to be meaningful?
3. Does the stack identify higher-risk buckets than baseline?
4. Is any apparent improvement stable by month and ETF category?
5. Is the result merely a mechanical reduction in trade count?
6. Is a costed replay justified before any further integration discussion?

## Promotion boundary

This phase cannot promote. A later phase would need:

- costed replay on actual trade/exit decisions;
- purged walk-forward or forward shadow validation;
- at least 20 new independent forward trading days;
- evidence that drawdown / sell efficiency improves without simply avoiding all
  trades;
- unchanged triple lock, SELL path, and risk gates unless separately approved.
