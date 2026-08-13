# Twelve-factor forecast plus point-in-time fundamentals V1

Status: `research-only / shadow-only / not trading`.

## Question

Can the four existing financial-statement mechanism families improve the calibrated
next-session judgement of the current complete twelve-factor model?

This is narrower than the earlier fundamental second-stage selector. The selected
Top10 is held fixed to the current guarded twelve-factor score. The only change is the
forecast input book:

- baseline: the same twelve completed price-factor ranks used by the current research
  forecast;
- candidate: those twelve ranks plus earnings innovation, growth acceleration,
  quality, and cash-flow-quality family ranks.

Each fundamental family remains the equal-weight composite preregistered in
`configs/research/fundamental_mechanism_families_v1.json`. No component, family or
weight is selected from historical performance.

## Causal financial-statement contract

- `noticeDate` is mandatory.
- A filing becomes usable on the first market session strictly after
  `max(noticeDate, updateDate)`.
- `reportDate` is used only for fiscal chronology and same-period comparison. It never
  determines availability.
- Every candidate row requires all four fundamental families. Missing fundamentals
  fail closed; the baseline is evaluated on the identical support.
- Price-factor completion is date-local and neutral, with at most two completed ranks,
  exactly as in the current research forecast.

## What is measured

The model family, training window, calibration window, regularisation and executable
next-open-to-next-open outcome are inherited without a parameter search. Baseline and
candidate are compared on identical stock-date rows for:

- gross-up Brier score, LogLoss, AUC and ECE;
- severe-loss Brier score, LogLoss, AUC and ECE;
- expected-return mean absolute error;
- the same metrics on the unchanged daily Top10; and
- paired day-level loss differences with Newey-West diagnostics.

Validation and shadow must both improve a majority of AUC/Brier/LogLoss for gross-up,
a majority for severe loss, and expected-return MAE both overall and inside the fixed
Top10. Probability dispersion by itself is never a pass condition.

## Interpretation boundary

All historical windows have already been viewed. This run can reject the increment but
cannot validate or promote it. A complete historical pass would only justify a separate
60-session fresh-forward preregistration. It would not change the dashboard, ranking,
orders, positions, risk gates, overlays, `build_decision()`, or any trading lock.

PBO is not applicable because there is one fixed challenger and no winner is selected
from a candidate menu. DSR is not applicable because the selected stocks and realised
return policy are unchanged; this experiment compares forecast losses. Previous
fundamental-integration attempts are nevertheless disclosed in the trial ledger.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_twelve_factor_pit_fundamental_increment_v1.py `
  --run-id run_YYYYMMDD_fixed_forecast_increment
```

Artifacts are isolated under:

`outputs/edge_research/twelve_factor_pit_fundamental_increment_v1/<run_id>/`

