# Perception-XAlpha Rolling Factor Health V4

## Why this study exists

The frozen twelve-factor book looked strong in its development and older validation
periods but reversed in the latest shadow window.  V3 showed that neither majority voting
nor a longer holding period repaired that instability.  V4 therefore tests one narrowly
preregistered mechanism: attenuate a factor only when its already-completed recent
cross-sectional IC no longer supports its original direction, and suspend the entire book
when its already-completed counterfactual results are unhealthy.

V4 does not add a factor, search another window, flip a factor direction or alter a base
weight.

## Causal sequence

For every day-t close:

1. Calculate the same twelve oriented factor ranks using data through t.
2. For each factor, use a 63-session rolling IC history with at least 40 observations.
3. Shift every IC by seven sessions before it can affect day t.  Seven covers next-open
   entry, the one-session hold and the fixed five-session delayed-exit allowance.
4. Multiply the frozen base weight by `clip(IC_t_stat / 2, 0, 1)` and renormalise positive
   weights.  Negative health can only turn a factor off; it cannot reverse it.
5. Require at least four active factors, then rank the adaptive Top10.
6. Maintain the Top10 counterfactual net return even when the book gate is closed.
7. Open the gate only when the lagged 40-completed-day history has at least 20 outcomes,
   a Beta(2,2) posterior net-win probability of at least 52%, positive rolling mean net
   return, and at least four active factors.

The gate therefore remains capable of reopening: closed days still generate shadow
counterfactual outcomes, but never orders.

## What is compared

- `static`: the unchanged twelve-factor weighted Top10.
- `adaptive_ungated`: rolling factor attenuation, always observed when enough factors live.
- `adaptive_gated`: the same adaptive Top10 only on days when the lagged health gate opens.

Every period reports signal coverage and names per day along with stock-level and
day-basket cost-positive win rate, Wilson lower confidence bound, mean net return, severe
loss rate and HAC t-statistic.  A gate cannot pass merely by selecting almost nothing.

## Research boundary

Validation and shadow windows have already been inspected.  They may reject V4 but cannot
promote it or tune another health window or gate threshold.  Even a pass would only justify
a separately preregistered fresh-forward shadow study.

The script has no broker, order, overlay, position, risk-gate or `build_decision()` path.

Run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_perception_xalpha_rolling_health_v4.py
```

Artifacts are isolated under:

`outputs/edge_research/perception_xalpha_rolling_health_v4/<run_id>/`
