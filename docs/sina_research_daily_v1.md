# Independent Sina daily price-basis recovery

This is a **research-only data collector**, not a new trading model or a fallback
inside the existing model. BaoStock data, frozen inputs, trading configuration,
orders, positions, gates, overlays and forward ledgers remain untouched.

## Source contract

Use Sina `hisdata_klc2/klc_kl.js` for unadjusted OHLC, daily volume **in shares**
and amount **in CNY**, and the same symbol's `hfq.js` for dated multiplicative
backward-adjustment factors. See the upstream [Sina interface documentation](https://akshare.akfamily.xyz/data/stock/stock.html#id29).
The bundled AkShare decoder is hashed in the run manifest. Only that local decoder
is executed: downloaded scripts are parsed as a single JSON assignment with optional
comments, not evaluated. No general `eval`, API key, proxy rotation or access bypass.

For each date `t`, take the last factor whose effective date is `<= t`:

```
adjusted_OHLC[t] = raw_OHLC[t] * factor[t]
raw_transaction_VWAP[t] = reported_amount_CNY[t] / reported_volume_shares[t]
adjusted_transaction_VWAP[t] = raw_transaction_VWAP[t] * factor[t]
```

Never apply today's factor to the entire past, infer factors from subsequent prices,
substitute OHLC4 for VWAP, silently change volume units, mix price providers or fill
missing bars. Missing factors, invalid OHLC/turnover and out-of-range transaction
VWAP produce null adjusted inputs with explicit rejection reasons. The fixed
engineering tolerance is 0.0001 CNY plus 0.001% of the raw high; it is not return-tuned.
Provider `prevclose`, `postVol`, `postAmt` are preserved as auxiliary raw metadata;
the script does not invent their inclusion/exclusion in aggregate turnover.

**Effective-date causal alignment is not proof of historical publication availability.**
This is the vendor's current vintage. Revisions and factor publication timestamps
are not reconstructed. Adjusted prices also are not a certified cash-dividend total
return series, and aggregate VWAP is not an executable price for a chosen order.

## Universe and missing metadata

Collect the union of the existing historical SH/SZ master (including known delisted
names) and current SH/SZ/BJ master. This does not solve all survivorship bias.
SH.689009 is a CDR with a different decoder and is excluded explicitly. Keep every
other requested symbol in the manifest even if its endpoint fails or history is
absent. Missing bars are counted between observed endpoints; that count cannot
distinguish suspension from provider omissions or certify pre-listing coverage.
Freshness and failed-symbol counts are reported separately.

Sina daily prices do not establish historical ST or suspension status: those fields
stay null, even for a numerically valid bar. A subsequent, separately audited loader
must join dated verified membership/status metadata without carry-forward, then
audit identical model support. **`readyForTraining` and `mayPromote` stay false.**
Three thousand fresh price rows only qualify the dataset for review, not training
or stock recommendations. No automatic model fitting or dashboard publication.

## Running

```
py -3.13 scripts/collect_sina_research_daily_v1.py --run-id pilot_YYYYMMDD --symbols SH.600000,SZ.000001,SZ.300750,SH.688981,BJ.920992
py -3.13 scripts/collect_sina_research_daily_v1.py --run-id full_YYYYMMDD
py -3.13 scripts/collect_sina_research_daily_v1.py --run-id full_YYYYMMDD --resume
```

Set `PYTHONIOENCODING=utf-8`. Defaults freeze 2019-01-02 through the **closed**
2026-09-08 session. The calendar is mandatory; no weekend-only fallback. A different
end date requires a new config contract/run. Minimum request spacing is two seconds,
one request at a time. Authentication/access denials, throttling or challenges stop
immediately; five consecutive transport failures stop the run. No retry storm.

The root-level OS lock prevents simultaneous collectors, including different run
IDs. Resume requires identical config, code, master, decoder and dependency hashes;
completed artifacts are checked before reuse. Failed symbol records are retained,
not silently retried. A new, explicitly reviewed run is required to retry failures.

Raw responses, raw bars, adjusted bars and the selected universe are written under
`data/market/ashare_research/sina_daily_v1/<run_id>/`. Audit manifest, per-symbol
records, progress/status and final result are under
`outputs/edge_research/sina_daily_v1/<run_id>/`. JSON/JSONL use atomic UTF-8 writes.
Each response/bar file is hashed. Old run directories cannot be overwritten.
Exit 0 means collection finished without symbol failures, **not** model validation;
exit 2 means blocked or completed with symbol gaps. Read the exact status/reasons.

## Validation / remaining work

`test_sina_research_daily_v1.py` tests the real run entry and resume path, response
injection rejection, past-only factor joins, no future-row impact, missing-factor
and unit errors, timezone/duplicates, missing sessions, unknown status, cache
tampering, single-instance locking, and access-denial handling. No network in tests.

After full collection: review daily price coverage, delisted/BJ failures, corporate
action discontinuities and independent cross-source overlaps; join audited dated
status; resolve known factor-formula/missingness issues under a new isolated model
version. Then consider the preregistered same-support Top10 experiment. Do not
rerun the exhausted price-volume factor search or claim better hit rates merely
because the source can now be downloaded.
