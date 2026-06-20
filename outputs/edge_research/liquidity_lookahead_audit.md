# Liquidity Look-ahead Audit

| Path | Look-ahead | Finding |
|---|---|---|
| select_t0_universe_live | False | uses the current snapshot cumulative amount and elapsed-session fraction only |
| fetch_yahoo_5m_quotes | True | fullamt[date] is computed from the complete day before the date's opening rows are emitted |
| convert_june_to_quotes | True | full-day amount selects the universe before all intraday decision rows are emitted |
| build_t0_replay_quotes_from_minute_data | False | liquidity threshold is evaluated from each timestamp's cumulative amount/session fraction |

## Measured overlap effect

- Raw ETF-days: 7978; selected by leaky gate: 4408.
- Mean open→close: all raw -0.1613%, leaky-selected -0.1537%, difference 0.0076%.
- This is not a valid correction to strategy PnL; the original full universe was not retained ungated.

## Decision

- treat prior baseline levels as method-contaminated diagnostics; rebuild before future opening/intraday claims.
- Live selector is unchanged; only future research must use the point-in-time path.
