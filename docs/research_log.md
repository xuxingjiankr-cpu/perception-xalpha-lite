# T0 ETF Quant — Research Log (tested hypotheses, verdicts, and why)

The institutional memory of what has been tried and what is **dead**, so no cycle (human or
LLM-agent) re-mines a known dead end. The moat is the GATE, not the generator: every claim
below was judged by purged out-of-sample + PBO (combinatorial backtest-overfitting prob) +
Deflated-Sharpe + day-clustered CI + REALISTIC cost (A-share ETF: 万三/side, no stamp duty →
万六 ≈ 6bps round-trip; book half-spread ~1–5bps/leg, 1 tick = 0.001元).

Status legend: ❌ dead (failed gate / cost-walled) · ⚠️ real-but-unusable · 🟡 open · ✅ live.

## Standing conclusions
- **Intraday 5-min A-share ETF timing/selection has no cost-surviving edge.** Real structure
  exists (short-term reversal) but is ~3–4 bps vs ~6 bps cost. Bottleneck = **edge/cost**, not
  noise, not tooling, not models.
- **Daily regime: cost stops binding, but no *stable* selection alpha** (momentum right-sign,
  t~2.2 recent, but non-stationary; PBO 0.5). This is the only **🟡 open** direction.
- **The A-share sixteen-factor Top10 book has no day-neutral selection edge** (#12), at any
  holding horizon from 1 to 20 sessions, under the frozen prior or equal weight. But it IS a
  strong, monotone, out-of-sample-stable ranker of SEVERE-LOSS risk at 1-5 sessions (#13). It is
  an exclusion tool, not a picking tool, and it is currently packaged as the opposite. The catch:
  what it really ranks is volatility, so the slice it flags as dangerous is also the slice that
  produced the best shadow-window return.
- The strategy's live buys net **+10 bps/trade at 万三** — but that's **beta**, not skill
  (day-neutral selection edge ≈ 0). Fine for a total-return competition; not alpha.

## Hypotheses tested this session
| # | Hypothesis | Method | Verdict | Why |
|---|---|---|---|---|
| 1 | Intraday momentum/breakout entry adds edge | 60d replay decision-quality | ❌ | Intraday momentum **reverts** (IC −0.12, t−22 OOS); buys = beta, no within-day selection (paired −0.39%) |
| 2 | Decision-score picks better ETFs within a day | day-paired + cluster bootstrap | ❌ | Pooled gap = market-day beta; same-day paired ≈ 0, CI incl. 0 (DSI-0008) |
| 3 | Sell logic loses to whipsaw → de-noise it (confirm/scale/breadth/multi-comp) | `ablate_sell_logic_v2.py`, 7 variants | ❌ | Cuts whipsaw but **destroys PnL** (+59k→−23k), missed-sell worse; **PBO=1.0**, DSR=luck. Whipsaw isn't free |
| 4 | Kalman de-lagged trend (Benhamou) beats raw momentum | `research_signal_ic.py` kf_velocity | ❌ | **Weaker** than raw momentum (\|IC\| 0.058 vs 0.123) — smoothing kills the reversal signal; wrong regime (we mean-revert) |
| 5 | Kalman residual (price vs trend) = tradeable reversion | kf_resid | ⚠️ | ≈ raw reversal (~0.11 IC), but **net −3.9 bps at 万三** |
| 6 | Short-term reversal is a real edge | `research_signal_ic.py` + cost sweep | ⚠️ | **Real & OOS-stable (t=12 full / 6.1 TEST)** but gross ~2–4 bps < 6 bps cost. Untradeable < ~2 bps cost |
| 7 | Drop anti-predictive momentum entry weights helps | `ablate_entry_signals.py` | ❌ | Per-trade quality ↑ (win 66% OOS, confirms IC) but **total PnL −32% DD**; PBO=1.0, DSR=luck |
| 8 | To-close relative-strength selector (daily signals) | `research_rs_selector.py` | ❌ | Fails even at **zero cost** (gross top-tertile negative); per-panel IC was a horizon artifact |
| 9 | Daily cross-sectional ETF rotation (mom/rev/vol), weekly | `research_daily_etf_rotation.py` 3y | 🟡 | Cost not binding; but excess sign-flips train→test, **PBO=0.5**, DSR=luck. Momentum t~2.2 recent but non-stationary. Underpowered (3y/5 factors) |
| 10 | Same-index ETF pairs long-only switch (KBPT, retail) | `research_pairs_switch.py` + spread sweep | ❌ | **Bid-ask-bounce mirage**: looks great at 0 spread (t=12, PBO=0) but breakeven < 1 bp/leg for the turnover-heavy configs; real ETF spread 1–5 bps/leg → dead |
| 11 | Order-book imbalance (OBI) is a harvestable intraday edge | `research_orderbook_imbalance.py`, 5.26M depth rows / 170 codes / 14d (06-25..07-31) | ❌ | **Misses the preregistered bar by ~10x.** IC is *negative* (+1: −0.0223, day-clustered t=−3.72); top-bottom spread 0.24–0.60 bps vs the **4.671 bps** measured median half-spread. Trade sim: TAKER −12.56 bps/trade (t=−810); even the OPTIMISTIC maker ceiling (ignores adverse selection) is −2.52 bps. Isolated paper account −1.584% / 13d. Audit: `outputs/l2_depth/obi_audit.md` |
| 12 | A-share Top10: reweighting or a longer horizon can clear the 30 bps cost | `research_horizon_cost_frontier_v1.py`, 6 horizons x 14 books, 2019-2026 | ❌❌ | **Two dead ends at once.** (a) *Weighting carries no information*: frozen prior vs EQUAL WEIGHT are indistinguishable at every horizon in both windows, and which one leads flips by horizon - four weighting studies (V1, V2, V3, walk-forward) were tuning a parameter that does not matter. (b) *Horizon does not help, and the apparent effect was beta*: gross swings from 11 to 247 bps (validation) and -16 to -302 bps (shadow) across h=1..20, but excess over the same-day eligible universe stays flat at -66..+16 bps with |t| <= 1.70 everywhere. Day-neutral selection edge ≈ 0 at every horizon, against a 30 bps cost. Same verdict the ETF line reached; now measured on stocks |
| 13 | The same book IS a stable tail-risk ranker even though it cannot pick winners | `research_tail_exclusion_screen_v1.py`, 10 deciles x 4 horizons x 2 books | 🟡 | **First gate pass in a long time.** Severe-loss rate rises monotonically across all ten score deciles (shadow h=1: 8.47% -> 24.98% against a 14.93% universe). Worst decile excess +12.5pp t=19.7 (validation) and +10.1pp t=11.8 (shadow); best decile -5.6pp t=-11.9 and -6.5pp t=-10.0. Survives at h=1 and h=5, dies by h=10 (shadow t=1.55) and h=20 (sign flips) - tail risk is stock-specific over days and washes into the market over weeks. **But it ranks VOLATILITY, not expected return**: the worst decile also had the HIGHEST excess return on shadow (+6.7 bps at h=1, +32.9 at h=5) while being negative on validation, so exclusion buys tail reduction at an unstable and possibly positive return cost. Frozen prior and equal weight agree to 4 decimals here too (#12). Historical windows viewed: fresh-forward only |

## Infra / data facts learned (don't re-discover)
- **Host TZ is KST (+09:00).** `minute_quotes.timestamp` is host wall-clock; use
  `source_quote_time` (Shanghai) for any time-alignment. Fixed in `decision_scoring.py`.
- akshare endpoints are sandbox-blocked; **Yahoo daily/5m works** (`*.SS`/`*.SZ`).
- Free intraday history is ~60d (Yahoo 5m); **free daily history is multi-year** — daily is the
  data-rich regime.
- **Absolute return is not evidence of selection at any horizon beyond a day or two.** In #12 the gross number moved 22x across the horizon grid while the day-neutral excess never left the noise band: at h=20 a 3% market move dwarfs anything the book picks. Always report excess over the same-day eligible universe, never gross, when the horizon is longer than one session.
- **The panel is cached** (`panel_cache.py`). Rebuilding it from 5443 jsonl / 5.5 GB took ~10-20 min and 4.5 GB; a cache hit is ~15 s. The rank book is cached the same way and is the next bottleneck after that. Cache misses by default on any key, checksum or format mismatch.
- **OBI is closed, not pending (see #11).** sina's FREE 5-level depth works and `collect_l2_depth.py` collected 14 clean sessions, but the signal misses the half-spread bar by ~10x. Don't re-open on more of the same free data; only true tick/queue (miniQMT) would be a different experiment.
- 131 same-index ETF "twin" groups exist (A500×15, 科创50×11, 沪深300×10 …) — natural pairs,
  but see #10.

## Tools built (all diagnostic/offline, gated)
`research_signal_ic.py` (IC/SNR + Kalman) · `research_decision_quality.py` (3-question replay)
· `research_rs_selector.py` · `research_daily_etf_rotation.py` · `research_pairs_switch.py`
(pluggable real spread via `--spread-file`) · `ablate_sell_logic_v2.py` · `ablate_entry_signals.py`
· `select_daily_momentum_pool.py` (nightly). Gates: `overfitting_guard.py` (PBO/CSCV/DSR),
purged CV, day-cluster bootstrap.

## Live changes (intentional, competition tilt — revert after July)
- ✅ Bold-play config (concentration, 95% deployment, let-winners-run exits) — variance-max for
  total-return tournament. `sell_logic_v2` shipped **off**.
- ✅ Nightly **daily-momentum top-20 pool** restricts the universe (correct horizon; fail-open).
  Revert = delete `dynamic_universe.daily_momentum_pool_file`.

## Real cost floor (measured)
- A-share ETF **half-spread ≈ 4.4 bps/leg median** (36,679 live bid/ask obs, 822 ETFs; liquid
  names 0.5–2 bps, illiquid up to ~9 bps). All-in round-trip aggressive ≈ commission 6 bps +
  spread ~9 bps ≈ **15 bps**; passive (maker) ≈ commission only. This kills the pairs switch at
  real spreads and raises the intraday wall.

## Open / in-progress directions (each must pass the gate)
1. **Daily, multi-year (~10y), many-factor, walk-forward rotation** (DDG-DA spirit for the
   non-stationarity). Cost not the wall here; stationarity is. Free data. Not yet run.

## The discipline (the actual moat)
No idea goes live without: purged OOS + PBO < 0.5 + Deflated-Sharpe beating the multiple-testing
noise band + day-clustered CI excluding zero + cost charged at realistic fills. Six tempting
ideas this session passed naive checks and **failed this gate.** That gate — and this log — is
what compounds into a better quant, not looping over empty data.
