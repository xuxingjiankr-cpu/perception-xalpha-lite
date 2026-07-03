# Laplace-Copula Same-Index ETF Price-Chain Replay

Status: `diagnostic_only / reused historical OOS / no live change`

- Train fit: 2026-03-23 to 2026-05-20.
- Exploratory test: 2026-05-21 to 2026-06-18.
- Universe: 12 benchmark groups, 37 ETFs.
- Frozen ordered pair models: 88.
- Laplace likelihood beat Gaussian in 67.0% of fitted marginal comparisons.
- Signal: leader up at least 0.20% over 15 minutes; lagger at least 0.15% behind and below the frozen 5% conditional tail; next-bar entry, 30-minute hold.

| Cost | Candidate return | Peer control | Paired edge | Sharpe | Trades |
|---:|---:|---:|---:|---:|---:|
| 6 bps | -0.52% | -0.81% | 0.28% | -4.712142517222588 | 39 |
| 12 bps | -0.99% | -1.27% | 0.28% | -8.595183665692216 | 39 |
| 20 bps | -1.61% | -1.89% | 0.28% | -12.901728069055766 | 39 |

## Evidence

- Paired daily bootstrap 95% CI: [-2.9998521810992425e-05, 0.00028653285402562196].
- DM vs cash: significant=`False`.
- DM vs peer: significant=`False`.
- DSR: significant=`False`.
- SPA: reject=`False`.
- Fresh unseen OOS gate: `False` (the test dates were already reused).

## Verdict

`no_validated_laplace_copula_price_chain_edge`

The frozen conditional-tail rule failed at least one return, cost, sample, paired-control or statistical gate.

A superior Laplace marginal fit only describes heavy tails; it is not evidence that the conditional event predicts a profitable catch-up.

## Paper applicability audit

- The 2018 Laplace-transform cash-flow article computes discounted present value and supplies no price-prediction mechanism.
- The 2022 paper motivates Laplace marginals plus a Gaussian copula, but reports one illustrative trade and ignores costs; this replay is therefore an independent long-only hypothesis test.
- arXiv:2607.01638 concerns Laplace-Beltrami PDEs for liquid crystals and has no defensible mapping to financial price chains.

## Limitations

- The 2026-05-21 to 2026-06-18 test window has already been inspected by other research and is not a clean untouched OOS window for this new hypothesis.
- Current benchmark membership can create survivorship bias.
- Yahoo five-minute prices and fixed round-trip costs do not reproduce order-book fills.
- The paper uses a long-short pair; this system is long-only, so the peer move is a control rather than a hedge.
- Laplace marginals are symmetric and the Gaussian copula has no non-degenerate tail dependence.
- A better marginal likelihood does not by itself establish predictive alpha.

This result cannot alter live entries, exits, sizing, overlays or execution locks.
