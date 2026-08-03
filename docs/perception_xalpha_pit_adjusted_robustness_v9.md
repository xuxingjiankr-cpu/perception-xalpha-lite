# Perception-XAlpha PIT-adjusted robustness V9

Status: **preregistered research-only robustness audit; not a new trading model**.

## Single question

Do the already frozen V2 stock ranks and V8 market-opportunity policy survive when the
known data defects are removed: raw corporate-action jumps, current-universe
survivorship, historical ST/suspension leakage and missing delisted fundamentals?

No factor expression, coefficient family, probability threshold, Top3 quota, holding
horizon, cost, walk-forward calendar or reliability threshold may change. The only
permitted change is the input universe in the preregistered override.

## Fail-closed prerequisites

The run must stop before any model fit unless all of these hold:

- adjusted-price audit covers at least 98% of the PIT master and passes every data check;
- the constructed panel reports adjusted prices, PIT membership and complete PIT
  ST/trading status;
- disclosure-date-aware fundamentals cover the selected PIT master without missing
  `NOTICE_DATE`;
- at least 3,000 symbols survive the frozen history and liquidity filters.

## Interpretation

A failure rejects robustness. A pass means only that the old frozen hypothesis survives
the corrected data; it does not create pristine OOS evidence because V1-V8 already
viewed the historical validation and shadow dates. Either result remains
`diagnostic_only`, writes no orders and cannot alter trading.

The post-backfill runner may wait unattended for the clean-data audit. It starts the
single frozen robustness run only after that audit passes and otherwise records a
blocked status:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts/run_perception_xalpha_pit_adjusted_robustness.ps1 `
  -BackfillRunId pit_full_20260803T2238
```
