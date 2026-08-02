# Perception-XAlpha Fundamental Discovery V5

Status: `research-only / shadow-only / not trading`.

## Scope

V5 extends the existing all-A-share OHLCV miner with point-in-time financial
indicators. It does not alter paper-trading configuration, `build_decision`, BUY/SELL
gates, position sizing, broker access, risk gates or execution locks.

The collector uses the Eastmoney financial-analysis endpoint exposed through AkShare.
Only rows carrying a real `NOTICE_DATE` are retained. A statement becomes visible to a
research signal on the first market date strictly after the later of `NOTICE_DATE` and
`UPDATE_DATE`. Report-period end dates never control availability.

## Feature groups

- Value: book-to-price and annualized YTD earnings yield.
- Quality: ROE, ROIC, gross/net margin and cash conversion.
- Growth: reported revenue and parent-profit year-on-year growth.
- Financial safety: debt/assets, current ratio and quick ratio.
- Operating efficiency: receivable days, inventory days and asset turnover.
- Interactions: quality x value, growth x momentum, quality x low beta, value x
  reversal, and safety x low beta.

The system still reports transaction-cost stress, Primary/Counter/Placebo tests,
purged walk-forward, PBO and DSR. Historical output cannot promote automatically.

## Known limitations

1. The security master is current rather than historical point-in-time. Delisted stocks
   and historical ST membership are incomplete, so survivorship bias remains.
2. The endpoint exposes currently retrievable statement versions. UPDATE_DATE delays
   availability conservatively, but unavailable old restatement vintages cannot be
   reconstructed.
3. Industry classifications are incomplete; this phase uses causal trailing-liquidity
   buckets, not true industry and market-cap neutralisation.
4. Annualizing YTD EPS is a transparent approximation and is evaluated separately from
   raw quality/growth factors.
5. A historical pass is only a forward-shadow hypothesis, never a trading signal.

## Automation

`scripts/run_perception_xalpha_fundamental_weekly.ps1` refreshes all statements,
audits disclosure dates, and runs an isolated 256-candidate V5 cycle. Isolated endpoint
failures are logged. The miner fails closed until coverage gates pass.

