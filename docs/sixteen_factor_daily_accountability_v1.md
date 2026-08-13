# Sixteen-factor daily accountability V1

This workflow is research-only. It does not place orders, alter positions, change the
sixteen-factor model, or promote a historical result.

## Why it exists

The sixteen-factor rank is always able to name ten stocks even when its probability
model has no useful directional conviction. That presentation can be mistaken for a
high-confidence trade list. V1 separates two outputs:

1. **Observation Top10** — always ten stocks, preserving the model ranking.
2. **Qualified shadow subset** — may contain zero stocks and is never an order.

The qualified subset uses fixed semantic thresholds that were not selected after the
2026-08-13 loss: `P(up) >= 50%`, expected gross return at least the 30 bp round-trip
cost, severe-loss probability at most 10%, and both frozen validation and shadow
directional reliability gates passing.

## Daily timing and labels

The Windows task `Eastmoney_Choice_Model_Top10_Daily` runs at 18:50 Korea / 17:50
Beijing, Monday through Friday. It:

1. updates the isolated PIT-adjusted SH/SZ archive;
2. retries only failed BaoStock shards serially;
3. creates the next-session sixteen-factor observation Top10;
4. refreshes the idempotent accountability ledger;
5. writes the supported Choice text-import watchlist.

The signal is observed after close on session `t`. The preregistered executable return
is next-session open to the following-session open, less 30 bp for the net result. The
entry-session open-to-close return is shown only as provisional information and is never
substituted for the final label.

## Outputs

- `outputs/edge_research/sixteen_factor_daily_accountability_v1/daily_ledger.jsonl`
- `outputs/edge_research/sixteen_factor_daily_accountability_v1/latest_daily_review.json`
- `outputs/edge_research/sixteen_factor_daily_accountability_v1/latest_daily_review.md`
- `logs/sixteen_factor_daily/sixteen_factor_daily_<date>.json`
- `data/research/eastmoney_choice_top10/Codex_Model_Top10.txt`

The daily ledger is rebuilt idempotently from immutable dated forecast artifacts and
market bars. A single losing day is recorded but never used for same-day weight fitting.
Any future model change requires a separately versioned, preregistered challenger and
fresh-forward evidence.
