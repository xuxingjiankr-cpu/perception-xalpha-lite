# A-share PIT adjusted data foundation V1

Status: **research-only / isolated / not a trading signal**.

## Why this exists

The existing all-A-share research dataset has two material limitations:

1. Its master is today's discoverable universe, so historical delistings and ST status
   are incomplete.
2. TDX/Sina OHLC is unadjusted. In addition, TDX volume is expressed in 100-share lots;
   treating it as shares made `amount / volume` (VWAP) approximately 100 times too large.

The loader now normalizes legacy TDX lots to shares before calculating VWAP. Future raw
collection writes an explicit `volumeUnit`, while old files remain byte-for-byte intact.

## Independent historical layer

`collect_ashare_pit_adjusted_baostock.py` creates a separate dataset and never overwrites
`bars_1d_raw` or the current master. BaoStock supplies:

- security type, listing date, delisting date and current/delisted status;
- backward-adjusted OHLC and adjusted previous close;
- daily trading status and historical `isST`;
- volume in shares and exchange-reported amount.

Only SH/SZ type-1 A-shares are admitted. BJ is excluded from the clean V1 rather than
mixed with current-membership data lacking equivalent historical ST coverage.

Backward adjustment (`adjustflag=1`) is used because its historical prefix is append
stable across later corporate actions. It corrects price discontinuities for return
research but is not represented as a cash-dividend total-return index.

## VWAP boundary

BaoStock amount/volume remains on the raw-price scale while adjusted OHLC is on an
adjusted scale. Mixing them would create another unit error. The clean layer therefore
stores `vwap` as an explicitly labelled adjusted OHLC4 proxy. Models must not interpret
it as true transaction VWAP.

## Eligibility

On each historical date a stock is eligible only when all are true:

- the date is within `[listingDate, delistingDate]`;
- `tradeStatus == 1` and volume/amount are positive;
- `isST == 0`;
- at least 120 prior observations exist;
- the shifted trailing 60-session median amount clears the frozen liquidity floor.

No value is forward-filled. Delisting, ST and suspension flags are part of the daily
eligibility mask, not whole-history filters inferred from the future.

## Quality gate

New factor research may reference this clean layer only after the audit confirms at
least 98% master-file coverage, at least 3,000 symbols with 500 observations, complete
adjustment/status metadata and invalid rows below 0.1%. Until then, existing research
continues to carry its survivorship/raw-price warning and no trading integration is
allowed.

## Full backfill

The full history can be collected in deterministic, disjoint shards without two
processes writing the same symbol:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ashare_pit_adjusted_backfill.ps1 -ShardCount 4
```

Each worker has an independent BaoStock session, stdout/stderr log and collection
summary. The coordinator waits for every worker and runs the fail-closed audit once at
the end. Parallel collection changes throughput only; it does not change the master,
eligibility rules or research quality gate.

The same PIT master can also be supplied to the existing disclosure-date-aware
fundamental collector. In resume mode it requests only master names whose statement
file is absent, including historical delistings omitted by the old current master:

```powershell
py -3.13 scripts/collect_ashare_fundamentals.py `
  --master-path data/market/ashare_research/baostock_pit_adjusted/master/ashare_pit_master_latest.jsonl `
  --output-root data/market/ashare_research/fundamentals_pit `
  --summary-name pit_adjusted_sh_sz_collection_summary.json `
  --workers 4 --resume
```

Disclosure rows without `NOTICE_DATE` are still rejected; no quarter-end date is
guessed. Fundamental coverage is measured against the selected PIT master rather than
against every unrelated file already present in the shared research directory.
