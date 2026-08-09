# Choice Model Top10 Watchlist

This integration publishes the latest frozen research Top10 as a stable, Choice-compatible
text file. It is a monitoring list, not a buy list and not an order source.

## Data path

1. The frozen shadow model writes `shadow_top10.json`.
2. `export_eastmoney_choice_top10.py` rejects any source that is not explicitly
   research-only, does not contain exactly ten unique SH/SZ A-share codes, or contains an
   order/trading mutation.
3. The exporter atomically replaces `data/research/eastmoney_choice_top10/Codex_Model_Top10.txt`.
4. Choice imports that fixed file into the dedicated self-stock group.

Choice private block files and private cloud endpoints are deliberately not modified. The
client's supported self-stock import flow remains the synchronization boundary.

## One-time Choice mapping

In Choice, rename the dedicated empty group `自选1` to `Codex_Model_Top10`, then map this
file through **Self-stock import -> Advanced options -> Add document**:

`C:\Users\XU XINGJIAN\Documents\Codex\data\research\eastmoney_choice_top10\Codex_Model_Top10.txt`

The scheduled exporter updates the same file after the daily research data refresh. If
Choice requires an explicit import/sync confirmation in the installed client version, keep
that client-side confirmation enabled; do not bypass it with private API calls.

## Manual commands

```powershell
py -3.13 scripts\export_eastmoney_choice_top10.py --dry-run
py -3.13 scripts\export_eastmoney_choice_top10.py
```

Use `--refresh-source` only after the A-share daily data collector has completed. On an
exchange holiday it leaves the previous valid list intact. If the market-data panel has
not reached the current trading session, generation fails before publishing and preserves
the prior valid import file.

The intended Windows task name is `Eastmoney_Choice_Model_Top10_Daily`, Monday to Friday
at 18:50 KST (17:50 Beijing time).
