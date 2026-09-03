# State-Conditioned Top10 Factor Experts V3

V2 asked which factors performed best over the latest seven fully resolved signal days. Its
historical improvement was too small and statistically weak. V3 tests a different mechanism:
which factors performed best during historical market states resembling the signal-day state?

For a portfolio intended for session `D`, every state feature is measured at the close of
`D-1`. Candidate analogue dates must be old enough that their entry, holding period and maximum
delayed exit were already completely known. No future state, return or label enters the distance.

The frozen state vector contains seven broad-market observations: five- and twenty-session
median-market return, twenty-session volatility, downside semivariance, breadth above the
twenty-session moving average, same-day cross-sectional return dispersion, and the median
stock-level amount shock relative to its prior twenty sessions.

For every signal date, V3 searches the previous 756 sessions for the twenty nearest fully
resolved states using a median/MAD scaler fitted on those past candidates only. It evaluates the
same four Top10 expert objectives as V2 and blends 70% state-analogue evidence with 30% global
63-session evidence. Weights remain positive, strongly shrunk to the frozen prior and bounded.

The method always selects exactly ten stocks. It is evaluated separately on future market-down
and market-up days, but those future labels never enter selection. Historical validation and
shadow windows are reject-only; the module cannot trade, promote, write overlays or alter any
production decision.

