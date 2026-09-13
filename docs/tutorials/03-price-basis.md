# 03 — One price field can change the whole rank order

[Run all cases](README.md) · [Generated report](../examples/audit-cases.md)

## Problem

For raw cash turnover and raw share volume, `amount / volume` is a raw cash price. Combining
it with adjusted close in `vwap / close - 1` mixes normalizations. Symbol-specific adjustment
histories do not cancel as a common cross-sectional scaling factor would.

## Minimal reproduction

```bash
python examples/run_audit_cases.py --output-dir outputs/audit-cases
```

Read `price_basis_inputs.csv` and `price_basis_comparison.csv`. Four fictional symbols have
adjustment multipliers 2, 1, 5 and 3. Their OHLC are adjusted; their cash amount and volume
are raw. The archive supplies a consistent **OHLC4 proxy**, explicitly not transaction VWAP.

## Wrong control → explicit basis guard

The wrong input substitutes raw amount/volume. The teaching adapter preserves the archive
field and checks declared basis metadata and same-bar OHLC bounds. A raw fallback against
adjusted OHLC raises `mixed_price_basis`; invalid archive values are not silently replaced.
These are explicit-fixture checks, not a universal price-quality classifier.

The largest proxy/cash ratio is 5.022545x and all four ranks change. Both arms use exactly
the same names; no eligibility filtering or data-source change is involved. A negative
control with consistently raw OHLC and raw cash VWAP is accepted.

## Interpretation and limits

Preserving OHLC4 repairs the fixture's basis contract but does **not** turn the proxy into
true VWAP. It does not establish which rank order predicts returns. The example is isolated
from the discovery loader; it does not claim the production data doctor already enforces
this new metadata check. Corporate-action vintages and execution-price accounting require
their own audit.

## Source → engineering test

QuantConnect's [corporate-action documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/corporate-actions)
distinguishes adjusted-price handling from raw-price cash-flow handling. This is a data
convention, not an alpha paper. The per-field basis check and synthetic counterexample are
our engineering demonstration; the source does not endorse this toy factor.
