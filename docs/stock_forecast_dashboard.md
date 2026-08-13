# Stock Forecast Observatory

`Stock Forecast Observatory` is a local, read-only research interface for the
daily A-share cross-sectional forecast. It deliberately separates presentation
from model implementation through the versioned
`stock_forecast_dashboard_v1` contract.

## Stable user contract

Every published snapshot exposes the same three estimates for each covered
security:

1. `probabilityUp`: estimated probability that the executable next-session
   return is positive.
2. `expectedGrossReturn`: estimated gross return over the registered forecast
   horizon.
3. `probabilityTailLoss`: estimated probability that the executable gross
   return is at or below -3%.

The page also exposes the factor rank, signal date, intended trading session,
estimate source and reliability status. Research models may be replaced or
extended, but their output must pass this adapter and contract validator before
the page accepts it. A breaking semantic change requires a new schema version;
it cannot silently redefine these fields.

The current model adapter requires one consistent twelve-factor calculation.
Limited missing ranks are completed with the same-date cross-sectional median
(a neutral rank, using no future row) in model fitting and prediction alike.
The page reports the completion count. Securities missing more than the frozen
support threshold are excluded rather than switched to a different model.

Return-amplitude calibration standardizes the raw training-period prediction
before applying the frozen Ridge calibration. This fixes the units mismatch
that previously collapsed most multivariate expected returns to one rounded
constant; it does not post-process or mechanically stretch the displayed
cross-section.

The snapshot also publishes the cross-sectional spread of expected return,
up-probability and tail-loss probability. A narrow calibrated probability range
is surfaced as weak discrimination; the interface never applies temperature
scaling or a multiplier merely to make probabilities look more decisive.

## Fundamental interaction shadow

For the exact same ranked Top10, the page may also display a research-only
fundamental interaction estimate. The four preregistered mechanisms are:

1. earnings innovation x abnormal trading amount;
2. growth acceleration x 20-session momentum;
3. accounting quality x low 20-session volatility; and
4. cash-flow quality x five-session reversal.

Fundamental availability is aligned by `noticeDate`, never by fiscal
`reportDate`. Market context uses only information available at or before the
signal date. The optional shadow fields are deliberately additive: they cannot
change rank, selected securities, factor score, orders, position, or any
trading gate. A date or Top10 mismatch fails closed.

The historical interaction hypothesis did not pass every preregistered gate.
Accordingly, the interface shows the adjusted estimates side by side for
diagnosis and labels them as shadow-only; the complete twelve-factor values
remain the stable primary contract.
Displayed deltas are measured against the primary values in the same dashboard
snapshot, while the interaction run's own baseline is retained as provenance.

## Sixteen-factor challenger ranking

The dashboard can alternatively be published from a full cross-sectional
reranking that combines the existing guarded twelve-factor score with all four
interactions. The frozen aggregation gives the twelve-factor block 75% and
each interaction 6.25%. This is equivalent to sixteen equal factor slots while
preserving the relative weights already used inside the guarded twelve-factor
block. None of these five block weights is fitted from historical outcomes.

Every ranked security must have all four PIT interaction ranks. Missing
fundamental interactions fail closed rather than being imputed. The resulting
Top10 is explicitly labelled `16 factors (12 + 4 interactions)` and remains a
research-only challenger: reranking the dashboard does not alter any order,
position, execution lock, risk gate or trading configuration.

## Open the page

Double-click `打开股票预测观察台.cmd` in the repository root. The launcher
starts a loopback-only server and opens:

```text
http://127.0.0.1:8765/
```

Search accepts a six-digit stock code, `SH.600000` / `SZ.000001` identifier,
company name, or pinyin initials such as `DHJS` for 鼎汉技术. Initials are
generated locally and never require an external API. The service binds to
`127.0.0.1`, so it is not exposed to the local network.

Install the small local transliteration dependency once when setting up a new
machine:

```powershell
py -3.13 -m pip install -r requirements-stock-forecast-dashboard.txt
```

## Daily publication

`generate_guarded_weight_top10_forecast_v1.py` publishes a dashboard snapshot
after it completes its normal forecast artifact. Existing results can be
published without rerunning the model:

```powershell
py -3.13 scripts/stock_forecast_dashboard.py publish --result <run-directory>\result.json
```

To generate the full sixteen-factor challenger ranking and publish it to the
read-only dashboard in one fail-closed job:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_stock_forecast_with_fundamental_interactions.ps1
```

An already generated same-date interaction artifact can be attached without
rerunning either model:

```powershell
py -3.13 scripts/publish_fundamental_interaction_shadow.py --interaction-result <run-directory>\result.json
```

Files are atomically written to:

```text
outputs/stock_forecast_dashboard/latest.json
outputs/stock_forecast_dashboard/snapshots/<intended-session>.json
```

## Local API

- `GET /health`
- `GET /api/latest`
- `GET /api/search?q=300011`

The service has no mutation endpoint. Every snapshot is marked
`research_only_read_only_not_trading`, contains empty `orders`, and is never
allowed to declare trading eligibility. It does not import broker, position,
order, overlay, or execution modules.
