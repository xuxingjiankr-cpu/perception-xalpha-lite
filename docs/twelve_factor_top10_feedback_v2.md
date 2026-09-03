# Causal Top10 Factor Feedback V2

## Research question

Can a frozen twelve-factor stock-ranking book improve its next-session Top10 selection by
learning from the factors that recently selected rising stocks, large relative winners and
fewer severe losers?

The module differs from the rejected guarded-online-weight V1. V1 adapted to rolling factor
IC. V2 treats every factor as a Top10 selection expert and scores the actual historical Top10
outcomes that the expert would have produced.

## Causal timing

For a signal at the close of session `t`, ranks contain information available at `t` only.
The hypothetical entry is the next tradable open. Outcomes are shifted seven trading sessions
before they may affect a later weight. This lag covers the entry, one-session holding horizon
and the maximum five-session delayed-exit allowance.

The recent feedback window therefore means the latest seven **fully resolved** signal days,
not the latest seven calendar rows and never unresolved future labels.

## Frozen feedback objective

Each factor expert selects ten stocks on every eligible day. Four equally weighted diagnostics
are calculated from its historical selections:

1. mean gross return;
2. fraction with positive gross return;
3. fraction belonging to that day's realised cross-sectional top five percent;
4. avoidance of returns at or below -3%.

The four diagnostics are ranked across the twelve experts so that return scale cannot dominate
the other objectives. Recent seven-day evidence receives 70% and trailing 63-day evidence 30%.
The resulting exponential tilt controls only 25% of the allocation around the frozen prior.
Every weight remains between 75% and 125% of its frozen value, remains positive and cannot
change by more than 4% total L1 weight per day.

## Weak-market requirement

The adapter never abstains and always uses exactly ten selections. Results are reported
separately for days when the eligible-market next-session return was negative and non-negative.
This tests whether the method can find relative winners on weak days instead of improving its
conditional statistics by refusing to trade.

## Decision boundary

The historical validation and shadow windows have already been inspected. They are reject-only.
Even a pass can justify only a separately preregistered fresh-forward shadow challenger.
The module cannot read or write trading configuration, create orders, call a broker, alter
`build_decision()`, change risk gates, size positions or publish a strategy overlay.

Run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_twelve_factor_top10_feedback_v2.py
```

