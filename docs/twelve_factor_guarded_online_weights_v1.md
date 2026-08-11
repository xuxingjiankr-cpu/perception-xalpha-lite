# Guarded Online Weights for the Frozen Twelve-Factor Book

## Decision

Automatic reweighting is allowed only as a research/shadow experiment. Earlier experiments
already rejected three less constrained approaches:

- a one-time twelve-factor utility fit failed its audit and both external profit gates;
- decision-focused weights improved the viewed validation block but deteriorated in shadow;
- 63-session rolling factor attenuation reduced or removed factors and did not improve both
  external periods.

Consequently, this version does not search another best set of historical weights. It asks
whether a deliberately low-freedom online adapter can make small, causal adjustments around
the existing frozen prior.

## Frozen rule

At each five-session update:

1. Shift factor IC observations by seven trading sessions. This covers next-open entry, the
   one-session hold and the maximum delayed-exit allowance.
2. Estimate each oriented factor's 20-session and 60-session IC t-statistics.
3. Use evidence only when both horizons have enough observations and agree in sign.
4. Cap evidence at +/-2 and discount it by the factor's mean absolute IC-history correlation
   with the other factors.
5. Apply a small exponential tilt to the frozen prior, then retain 75% of the prior and only
   25% of the tilted allocation.
6. Keep each factor between 75% and 125% of its frozen weight. No factor can become zero and
   no factor direction can flip.
7. Cap the L1 change of one update at 8%. If evidence is insufficient, use the frozen prior.

This is an online algorithm, not an expanding parameter search. Validation and shadow results
may reject the algorithm but cannot change any window, cap or coefficient.

## Fair comparison

Static and adaptive books use:

- the same PIT-adjusted all-A-share panel;
- the same twelve factor definitions and directions;
- the same executable next-open to next-open outcome;
- the same 30-basis-point round-trip cost;
- the same daily candidate support and exactly ten selections;
- no market-timing gate and no abstention.

The report includes the paired daily return difference and Top10 overlap. Thus the adapter
cannot appear better merely by trading less often.

## Research boundary

All historical windows have already been viewed. Even simultaneous improvement in validation
and shadow would only justify a separately preregistered fresh-forward shadow ledger. A failed
window means the static frozen prior remains the default; it does not authorise a search over
3-day, 7-day or alternative update windows.

The module cannot read or write trading configuration, create an order, call a broker, alter
`build_decision()`, change a risk gate, size a position or write a strategy overlay.

Run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_twelve_factor_guarded_online_weights_v1.py
```

Artifacts are isolated under:

`outputs/edge_research/twelve_factor_guarded_online_weights_v1/<run_id>/`
