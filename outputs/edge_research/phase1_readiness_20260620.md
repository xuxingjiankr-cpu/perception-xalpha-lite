# T+0 ETF Edge Research — Phase 1 Readiness

Generated: 2026-06-20T10:37:42+08:00

Research only. No live configuration or execution path is changed.

## T+0 master

- Total rows: 1538
- Officially confirmed T+0: 165
- Confirmed and present in local non-money universe: 131
- Confirmed with 20 turnover observations: 84
- Explicit T+1: 695
- Pending product verification: 654
- Money-like excluded: 24
- Limitation: SZSE public list lacks a product-level T+0 class; SZ names remain fail-closed pending verification

## Research directions

| # | Direction | Status | Evidence |
|---:|---|---|---|
| 1 | watchlist effectiveness | blocked | 0 real watchlist days; need point-in-time history |
| 2 | T+0 ETF master | partial | 165 confirmed; 654 pending |
| 3 | cross-market lead signals | blocked | no timestamp-aligned futures/FX/yield series |
| 4 | premium/discount filter | blocked | no point-in-time IOPV/NAV series |
| 5 | opening 30 minutes | partially_ready | 60 Yahoo 5-minute days; no validated watchlist split |
| 6 | trend vs mean reversion | partially_ready | 60 days; only officially classified SH instruments can be grouped safely |
| 7 | liquidity/slippage | partial | 20-day turnover available for some instruments; historical spread/depth unavailable |
| 8 | news decay | blocked | 0 real watchlist days |
| 9 | ETF relative strength | partially_ready | price/turnover usable; theme, premium and news relevance history absent |
| 10 | kill switch | partial | daily-loss and execution safeguards exist; premium/news/data-source gates require missing inputs |

## Next data actions

- resolve SZ product-level categories from official fund announcements or a documented exchange field
- accumulate at least 20 point-in-time ChatGPT watchlist trading days
- collect real bid/ask/depth and IOPV/NAV rather than synthetic replay values
- add timestamp-aligned external futures, FX, yields and commodity references
