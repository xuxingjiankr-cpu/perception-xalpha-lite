# Point-in-Time Data Doctor

`xalpha-doctor` is a fail-closed preflight check for the two tables consumed by the public
factor-research pipeline. It answers a narrower question than a backtest:

> Can these files enter this research protocol without violating a declared data contract?

It does not score a factor, estimate a return, or make a trading decision.

## Run it

```bash
xalpha-doctor \
  --prices data/prices.csv \
  --fundamentals data/fundamentals.csv \
  --config configs/example.json \
  --output outputs/data_readiness.json
```

The default exit policy is suitable for CI:

| Exit code | Meaning |
|---:|---|
| `0` | no blocking violation; warnings may remain |
| `2` | at least one blocking contract violation |
| `3` | warnings remain and `--strict-warnings` was requested |

## Checks

| Contract | What is inspected | Why it fails closed |
|---|---|---|
| Price schema | date, symbol, OHLC, volume, amount | The panel cannot be reconstructed without an explicit bar contract |
| Unique keys | one row per date-symbol | Duplicate bars make pivot and return construction ambiguous |
| Bar integrity | OHLC ordering, positive prices, non-negative activity | Corrupted bars can manufacture ranks and fills |
| Chronological capacity | train + purge + validation + purge + shadow | A split that does not fit is not a validation design |
| Neutralization support | industry and point-in-time market cap | A declared neutral portfolio cannot silently become an unneutralized one |
| Tradability support | ST, suspension, delisting, limit-up and limit-down | A quote is not evidence that a trade was executable |
| Disclosure timing | report, notice and update dates | `report_date` is not an information-availability timestamp |
| Feature payload | numeric fundamental fields | Metadata without an aligned payload cannot form a factor |
| Future-named fields | target/label/forward/outcome patterns | Presence is surfaced for review; the name alone is not mislabeled as proof of leakage |

The JSON report includes deterministic input digests, every individual check, summary counts,
and an empty `orders` array. It can be committed beside a run manifest as the data-side portion
of a research birth certificate.

## Deliberate non-claims

A CSV cannot prove its own provenance. The doctor therefore leaves explicit warnings for:

- historical security-master membership and survivorship;
- vendor restatement and revision history;
- price-adjustment and corporate-action convention;
- economic validity or predictive value of a field.

A clean report means the declared interface is internally consistent. It is not evidence of
alpha, profitability, or deployability.
