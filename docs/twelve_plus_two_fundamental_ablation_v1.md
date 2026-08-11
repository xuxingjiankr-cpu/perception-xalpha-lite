# Frozen twelve-factor model plus two fundamental factors

Status: `research-only / shadow-only / not trading`.

## Question

Do the two closest candidates from the financial-statement discovery run improve the
frozen twelve-factor model's next-session Top10?

The frozen additions are:

- inventory turnover days level, oriented so fewer days rank higher; and
- operating cash flow divided by revenue, oriented upward.

These candidates were chosen after inspecting a 64-candidate historical run.  This is
post-selection evidence, not a new clean holdout.  The study charges all 64 prior trials
plus three new extension policies and can only reject the combination.

## Frozen weights

The original twelve-factor weights remain unchanged relative to one another.

- Baseline: 100% frozen twelve-factor meta-score.
- Inventory ablation: 12/13 price score and 1/13 inventory-days score.
- Cash ablation: 12/13 price score and 1/13 cash-to-revenue score.
- Requested combination: 12/14 price score, 1/14 inventory-days score and 1/14
  cash-to-revenue score.

No historical outcome selects or refits these weights.

## Fair comparison

Every policy uses the same price-factor Top100, requires both fundamental values, and
selects the same daily Top10 count.  The price-only baseline is reranked on that identical
complete-support universe.  Consequently, missing financial statements, reduced
coverage or fewer selections cannot mechanically create improvement.

The outcome is formed after the signal-day close, enters at the next buyable open and
exits at the following sellable open.  Both gross and 30-basis-point costed metrics are
reported.  Since all policies trade the same count, their paired return difference is
unchanged by the common cost assumption.

## Required evidence

The requested two-factor policy must improve, in both validation and shadow:

- stock-level gross-up rate;
- stock and daily mean gross return; and
- severe-loss rate.

The validation paired-return p-value is compared with a family-wise threshold that
charges 67 total trials.  Even a full historical pass would require a separate 60-session
fresh-forward preregistration.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_twelve_plus_two_fundamental_ablation_v1.py
```

Artifacts are isolated under:

`outputs/edge_research/twelve_plus_two_fundamental_ablation_v1/<run_id>/`

The script cannot read or modify trading configuration, daily Top10 artifacts,
`build_decision()`, BUY/SELL gates, positions, orders, overlays, risk gates or execution
locks.  Every result contains `orders: []`.
