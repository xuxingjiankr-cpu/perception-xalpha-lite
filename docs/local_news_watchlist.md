# Local ETF News Watchlist

Status: `diagnostic_only / paperTradingOnly / no trade gate`

## Purpose

`scripts/generate_local_news_etf_watchlist.py` replaces the unavailable hosted-model
watchlist with a small local evidence-ranking pipeline. It uses public Google News RSS,
the previous completed Eastmoney ETF snapshot, and a versioned deterministic score.
It does not require an API key and does not call OpenAI or another hosted model.

The output is an observation list, not a BUY list. No score is represented as a
probability.

## Point-in-time contract

- News window starts at 15:00 China time on the previous XSHG session.
- The data cutoff is the actual generation time, normally 08:30 China time.
- Articles after the cutoff are rejected.
- Liquidity and market confirmation use only the previous completed session snapshot.
- Historical `--as-of-date` runs are dry-run only and cannot overwrite the forward ledger.

## Score

The checked-in `LNW-1.0.0` score is:

```text
0.45 * news_direction
+ 0.15 * news_coverage
+ 0.10 * source_quality
+ 0.15 * previous-day_liquidity
+ 0.15 * previous-day_market_confirmation
- crowding_penalty
```

All components, source headlines, source timestamps, publisher names, and URLs are
stored with each selection. `evidenceConfidence` measures evidence coverage and
agreement; it is not a calibrated success probability.

## Files

- Config: `configs/research/local_news_watchlist_v1.json`
- Daily inbox: `data/research/local_news_etf_watchlist/inbox/<date>.json`
- Raw evidence: `data/research/local_news_etf_watchlist/raw/<date>/articles.jsonl`
- Logs: `logs/local_news_etf_watchlist/local_news_etf_watchlist_<date>.log`
- Observation pool: `outputs/t0_observation_pool/latest_observation_pool.json`

## Schedule

Task Scheduler name: `Local_ETF_News_Watchlist_Daily`

The task runs Monday through Friday at 09:30 Korea time, equivalent to 08:30
China time. The legacy `ChatGPT_ETF_Watchlist_Daily` task is disabled by the
installer unless explicitly retained.

## Validation status

The generator and receiver have invariant coverage for:

- cutoff-safe RSS parsing;
- positive and negative evidence retention;
- exactly ten unique non-money ETFs;
- local source provenance;
- research-only safety markers;
- absence of hosted-model and order submission paths.

The score has no validated return edge. It must accumulate forward outcomes before
any weighting changes are considered. The appropriate first test is high-score versus
low-score next-session return after costs, clustered by independent trading day.

## Literature boundary

- NBER w34965 supports using model token probabilities instead of model-declared
  confidence. This pipeline has no language model, so it does not claim to reproduce
  that result.
- arXiv:2606.23492 supports heavy-tailed HMMs for synthetic scenarios and conditional
  VaR. It does not demonstrate a trading-return edge and is not part of this ranker.
- arXiv:2606.26804 and arXiv:2606.02118 are estimation-method papers, not trading
  signals.
- arXiv:2606.22601 is an astronomy anomaly-detection application; arXiv:2606.19430
  is quantum many-body dynamics. Neither supplies financial out-of-sample evidence.
