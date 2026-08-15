# Event-first opening-auction forward study V1

> Status: research-only, shadow-only, not trading.

## Frozen question

The study does not search another price-volume factor and does not retrain the
existing twelve- or sixteen-factor books. It prospectively records two fixed
arms beginning strictly after 2026-08-14:

1. `event_liquidity`: stocks with a positive, causally available filing change
   on the immediately preceding market session. Trailing 20-session amount is
   used only as an execution tie-break when more than ten names qualify.
2. `event_auction_confirmed`: the same pool, but only on mornings when more
   than half of the observed market opens above the previous close and the
   median opening gap is non-negative. A stock must also have a positive
   opening gap.

Both arms may select zero to ten stocks. There is no daily quota. A zero-name
day is an intended observation, not a failure.

## Causal clock

- Filing availability remains the existing rule: the first market session
  strictly after `max(noticeDate, updateDate)`.
- The event must already have completed its safe session before it can become
  a next-session opening candidate.
- The opening snapshot must have a source quote time between 09:25:00 and
  09:29:59 Asia/Shanghai and contain at least 3,000 A-share rows.
- Same-session close is an outcome only. It cannot revise the morning file.
- Historical and forward outcomes cannot tune this version. A parameter change
  requires a separately preregistered version.

## Controls and interpretation

Each arm receives two controls. A deterministic same-pool, same-count
permutation tests whether the liquidity tie-break itself appears predictive. A
second control selects non-event stocks under the same date and auction state,
matched without replacement on prior log-ADV. The second control is the primary
test of event-pool membership. This separates a useful event pool from an
apparent rank effect. The earlier catalyst study showed exactly why this
matters: recent event windows had attractive absolute returns, while the
proposed catalyst rank did not beat same-pool random selection.

A verdict is forbidden before twenty independent resolved active days; sixty
days is preferred. Accuracy is always reported with coverage so that selecting
fewer names cannot masquerade as a free improvement. A positive verdict must
simultaneously improve stock-level win rate and mean gross return without
raising the severe-loss rate relative to the matched control.

## Outputs

The runner writes only beneath:

`outputs/edge_research/stock_event_auction_forward_v1/`

Daily morning predictions and post-close outcomes are stored separately and
rolled into append-style JSONL ledgers. Every artifact contains `orders: []`.
No trading config, overlay, order, position, risk gate, sizing rule, execution
lock, observation pool, or production decision path is read or modified.

## Commands

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/run_stock_event_auction_forward_v1.py --mode score
py -3.13 scripts/run_stock_event_auction_forward_v1.py --mode settle
py -3.13 scripts/run_stock_event_auction_forward_v1.py --mode report
```

`score` fails closed if the prior daily panel, fundamental coverage, auction
timestamp, or market-row requirement is not satisfied. `settle` never changes
the morning prediction and waits until the trade-date daily bar is available.

The installed Windows research tasks run the wrapper on weekdays at 10:32 KST
(09:32 Asia/Shanghai, after the frozen-window auction snapshot) and 18:30 KST
for settlement. Running on an exchange holiday creates no prediction because
the required date-stamped auction snapshot is absent; this is a fail-closed
diagnostic, not a trading action.
