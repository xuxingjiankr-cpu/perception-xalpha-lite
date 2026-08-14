# Opening-auction signal amplification V1

## Research question

The frozen sixteen-factor model is evaluated after the prior session closes.  It
cannot observe news arriving overnight or the next opening auction.  This study
tests whether the first new cross-sectional price information at 09:25 improves a
Top10 chosen for the same session's open-to-close return.

The study has only two challengers: one global pairwise ranker and the same ranker
with four causal market-state routes.  This is not a search over blend weights,
lookbacks, routes or thresholds.

## Primary research support

- Mandi et al., *Decision-Focused Learning: Through the Lens of Learning to Rank*
  (ICML 2022), motivates fitting the ordering of feasible decisions rather than a
  point forecast that may not preserve the Top10 ordering:
  <https://proceedings.mlr.press/v162/mandi22a.html>.
- Lin et al., *Learning Multiple Stock Trading Patterns with Temporal Routing
  Adaptor and Optimal Transport* (2021), motivates separate predictors under
  non-stationary stock patterns; this implementation uses a deliberately shallow,
  auditable router instead of importing the neural architecture:
  <https://arxiv.org/abs/2106.12950>.
- Jiang and Li, *Adverse Selection and Overnight Returns: Information-Based
  Pricing Distortions Under China's T+1 Trading* (2025), links negative overnight
  returns and opening-session recovery under A-share T+1 constraints:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5349222>.
- Chung et al., *An Empirical Analysis of the Shanghai and Shenzhen Limit Order
  Books* documents return predictability from Chinese order imbalance and
  motivates a later auction/L2 data phase if this daily-bar feasibility test is
  positive: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2055179>.

These papers motivate hypotheses; their reported results are not evidence that
this repository's data contains a usable edge.

## Causal timing

For execution session `d`, the prior-close score is shifted from `d-1`.  The only
new same-session inputs are the opening price, point-in-time listing/ST/trading
status and cross-sectional functions of the opening gap.  Same-day high, low,
close, volume and amount are forbidden from the feature and selection masks.
Same-day close is read only as the offline label.

The daily open is the auction clearing price and cannot be guaranteed after the
auction is observed.  A 30 bp round-trip friction is therefore shown, but cannot
repair queue-priority or 09:30 slippage uncertainty.  A positive result can only
justify collection of real auction snapshots and first-minute execution prices.

## Acceptance

Against an auction-time control on exactly the same feature-complete universe, a
challenger must improve Top10 win rate by at least one percentage point and mean
gross return by at least five basis points, improve mean cross-sectional return
percentile, not worsen the severe-loss rate, and improve both win rate and gross
return in a majority of fixed walk-forward blocks.

All historical windows are already viewed.  The result is research-only,
shadow-only, non-trading and cannot promote itself.

## Recorded V1 result

`run_20260814_auction_signal_amplification_v1` rejected both challengers.  The
auction-time prior-close control had a 51.59% win rate, 0.0413% mean gross return
and 3.22% severe-loss rate.  The routed pairwise winner ranker raised mean gross
return to 0.1566%, but reduced the win rate to 50.44% and raised severe losses to
9.22%.  It learned high-dispersion lottery exposure rather than accuracy.
