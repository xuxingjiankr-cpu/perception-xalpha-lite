# Perception-XAlpha V10: clean PIT factor evolution

Status: `research_only / shadow_only / not_trading`.

V10 is the first autonomous factor-search ledger that is allowed to use the isolated
BaoStock SH/SZ panel with backward-adjusted OHLC, listing/delisting intervals, historical
ST flags and historical trading status. It deliberately starts with an empty parent pool.
Parents selected on the current-universe/raw-price V4/V5 data are not migrated because
their fitness is not comparable to the clean panel.

## Frozen first-cycle design

- Input: all available SH/SZ A shares whose historical membership overlaps 2019 onward,
  including delisted stocks supplied by the clean master.
- Features: the existing past-only OHLCV/turnover and point-in-time fundamental DSL.
- Label: next-open to the open ten trading sessions later, subject to causal membership,
  ST, suspension and locked-limit tradability masks.
- Candidate budget: 256, at most 24 Stage-2 bundles, eight train-only generations.
- Search feedback: train fast-screen metrics only. Validation and shadow metrics are never
  returned to the generator or parent selector.
- Controls: Primary, economic Counter, causal delayed Placebo, purged walk-forward, PBO,
  project-wide DSR and separately reported 30 bps round-trip cost stress.
- Baseline historical trial count: 1,669. Every completed V10 cycle adds its generated
  candidate count automatically; the multiple-testing burden never resets.
- Trigger: a changed input fingerprint only. Re-running unchanged data returns
  `no_new_data` rather than manufacturing another trial.

The first clean cycle intentionally keeps the V5 grammar, thresholds, label horizon and
book construction unchanged. This makes its result a data-quality comparison rather than
another parameter search. A threshold change requires a new preregistered config version.

## Continuous evolution semantics

Each completed clean cycle may retain at most 20 train-selected parents in the isolated
V10 state directory. When genuinely new input data arrive, the novelty epoch changes and
the next cycle performs bounded mutation, crossover, refinement and novelty injection.
The engine may discover zero credible factors; zero is a valid scientific result and the
target count of five is never forced.

`BORN` means a historical research candidate survived the configured controls. It does
not mean tradeable, profitable or approved. Historical output cannot modify a strategy,
write an overlay, call a broker, create an order, alter risk gates or promote itself.
Separate fresh-forward evidence and explicit human approval remain mandatory.

## Fail-closed data gate

Before any model fit, V10 requires:

1. exactly the isolated SH/SZ PIT-adjusted data root;
2. at least 98% master-file coverage and at least 3,000 eligible stocks;
3. adjusted prices for every accepted file;
4. historical ST and trading-status fields for every accepted row;
5. listing dates and point-in-time membership for every accepted master record;
6. point-in-time fundamental availability using the first market date strictly after
   `max(NOTICE_DATE, UPDATE_DATE)`.

Failure blocks the cycle. It never falls back to the old raw/current-universe dataset.

## Known limitations

- This is historical research, not pristine future OOS; the historical interval has been
  inspected in earlier research.
- BaoStock backward adjustment is not a cash-dividend total-return index.
- Historical financial-statement restatement vintages can still be incomplete.
- BJ is excluded because equivalent clean historical membership/status coverage is absent.
- Size neutralisation uses trailing liquidity buckets; industry classification is not yet
  point-in-time and is therefore not fabricated.
- No factor, model or process can guarantee profit, win rate or a profitable Top 3.
