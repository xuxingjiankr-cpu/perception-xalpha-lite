# Financial-statement factor discovery V1

Status: `research-only / shadow-only / not trading`.

## Research question

Can a bounded, point-in-time financial-statement grammar discover accounting factors
that improve the next-session Top10 gross-up probability and return after the frozen
price-factor model has selected its Top100?

This is not the earlier fixed 16-indicator score.  The generator expands an audited
field registry into candidate formulas before reading return labels.  It creates:

- profitability levels and issuer changes;
- reported growth levels and growth acceleration;
- same-fiscal-period symmetric growth for EPS, revenue, parent profit and book value;
- cash-flow confirmation and cash improvement;
- leverage, liquidity and solvency levels and changes;
- receivable, inventory and asset-turnover efficiency levels and changes;
- within-family breadth and strict-confirmation composites; and
- cross-statement confirmations such as growth plus cash, growth plus margin, and
  efficiency plus margin.

Every candidate receives an immutable formula hash, mechanism family, input fields,
maximum age and economic hypothesis.  The grammar is deterministic, label-free and
bounded to 96 generated candidates and 24 full evaluations.

## Causal timeline

1. A filing is unavailable until the first market session strictly after
   `max(noticeDate, updateDate)`.
2. If one API retrieval exposes several historical reports on the same safe date, the
   latest report is the event and older reports are only causal comparison context.
3. A non-advancing late report is skipped because the unavailable original restatement
   vintage cannot be reconstructed.
4. `reportDate` is used only for fiscal chronology and same-period comparisons.  It never
   determines market availability.
5. The score is formed after the safe disclosure session closes.  Entry is the next
   buyable open and the V1 outcome is the following sellable open.
6. Outcome labels never enter the event table or generator.

## Selection and evaluation

The frozen 13-price-factor model first selects exactly 100 candidates.  Each accounting
candidate is neutralised inside five causal trailing-liquidity buckets and receives a
fixed 20% weight; the frozen price rank keeps 80%.  This fixed weight is a single
preregistered ablation, not a searched optimum.

For every factor, its blended Top10 and the price-only Top10 use the identical complete
support and identical daily selection count.  Therefore a gain cannot be manufactured by
removing difficult stocks or trading less often.

Only train-period IC and paired daily return can choose the 24 candidates sent to full
evaluation.  Validation and shadow metrics do not feed synthesis, mutation, weights or
candidate selection.  The complete generated count is charged to the Benjamini-Hochberg
multiple-testing threshold.

Reported metrics include:

- Top10 gross-up and costed win rates;
- mean and median gross/net return;
- severe-loss rate and daily CVaR;
- Spearman IC with Newey-West t-statistic;
- paired daily return against the same-support price baseline; and
- validation multiple-testing result.

A historical candidate must improve both gross-up rate and mean gross return, have
positive IC in validation and shadow, and pass the generated-trial correction.  Even a
full historical pass cannot trade.  It only permits a separately frozen 60-session
fresh-forward hypothesis.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_financial_statement_factor_discovery_v1.py
```

Artifacts are written atomically beneath:

`outputs/edge_research/financial_statement_factor_discovery_v1/<run_id>/`

The directory contains the candidate birth registry, train fast screen, full
validation/shadow result, report and run manifest.  Every result contains `orders: []`.

## Limitations

- Historical validation and shadow windows have already been viewed by other studies.
- The provider may not expose every original historical restatement vintage.
- Historical industry membership is unavailable.  Liquidity buckets reduce a size bias
  but do not provide full industry neutralisation.
- Financial statements are slow information.  Their most plausible next-session role is
  risk filtering or post-disclosure confirmation, not minute-level timing.
- Daily bars cannot reproduce opening-auction queue priority.

The module cannot read or modify trading configuration, the observation pool,
`build_decision()`, BUY/SELL gates, positions, orders, risk gates, overlays or execution
locks.
