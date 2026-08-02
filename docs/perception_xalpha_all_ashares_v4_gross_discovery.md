# Perception-XAlpha all-A v4 gross factor discovery

Status: `research_only / shadow_only / not trading`.

## What changed

V4 separates two questions that must not share a rejection gate:

1. Does a past-only expression predict the future cross-section before implementation
   costs? This is the **factor-discovery objective**.
2. Can a portfolio implementation retain that return after conservative costs? This is a
   separate **tradeability stress test**.

The 30 bps stress test remains in every artifact, but a negative costed IR no longer hides
a statistically persistent gross factor. No cost is set to zero and no metric is deleted.

## Daily output tiers

- `daily_factor_candidates.json`: at most five diverse gross candidates that passed the
  Primary/Counter/Placebo, purged walk-forward and PBO controls before project-wide DSR.
  These are hypotheses requiring new data, not trading factors.
- `credible_research_factors.json`: the stricter subset also surviving project-wide
  multiple-testing/DSR. The target of five is never forced; zero is a valid result.
- `factor_bundles.json`: complete train/validation/shadow, gross/net, turnover and cost
  evidence for every Stage-2 candidate.

## Search budget and independence

- Up to 256 new expressions and 24 Stage-2 bundles per completed cycle.
- Seven economic mechanism families and a bounded expanded OHLCV seed library.
- Train-only evolution; validation and shadow never return to the generator.
- Behaviour correlation above 0.85 is rejected to avoid reporting five cosmetic variants
  of the same exposure.
- Five purged walk-forward folds with a ten-trading-day purge.
- The project-wide trial ledger starts at 901, conservatively including the earlier
  historical probes and the already-launched superseded v2 batch.

## Known limitations

The master is not point-in-time and the historical window has been repeatedly researched.
Consequently, historical survivors cannot promote. A useful candidate must be frozen and
validated on new forward data before any separate discussion of portfolio construction or
paper-trading use.
