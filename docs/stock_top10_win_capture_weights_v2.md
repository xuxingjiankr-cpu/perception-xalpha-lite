# Stock Top10 win-capture weight audit V2

## Question

Can positive, bounded, rolling weights on the current sixteen-factor book, direct
point-in-time fundamental families, or their price interactions improve the
next-session Top10 win rate and gross open-to-close return at the same time?

The target is fixed before fitting: a signal recorded after session `t` closes is
evaluated on session `t+1` as `close / open - 1`.  The Top10 is ranked before its
future outcome or future tradability is read.  A selected name with an unresolved
outcome is retained as unresolved and is never replaced by the eleventh name.

## Protocol

- 504-session trailing fit window, 63-session refit interval and 10-session purge.
- Pairwise winner objective: daily positive Top10 observations are contrasted with
  near-zero non-positive observations and the worst losses.
- Every factor keeps a positive weight between 1% and 15%; no factor is silently
  eliminated.
- Every policy selects exactly ten intended names per day.
- Acceptance requires simultaneous improvement in win rate, mean gross return,
  cross-sectional percentile and true market-Top10 overlap, no worse severe-loss
  rate, and improvement in a majority of walk-forward blocks.
- Invalid full-history hindsight weights are reported only as an overfit-exposure
  diagnostic and are never published as a usable model.

## Historical result

The corrected intraday run is
`run_20260814_intraday_win_capture_v2`, using data from 2019-01-02 through
2026-08-13.  The frozen baseline reached a 51.57% gross win rate and 0.0476% mean
gross open-to-close return.  No challenger passed:

| Candidate | Win rate | Mean gross | Severe loss | Decision |
|---|---:|---:|---:|---|
| frozen sixteen-factor baseline | 51.57% | 0.0476% | 3.13% | reference |
| reweighted current sixteen | 50.60% | 0.0704% | 5.45% | reject |
| twelve price plus four direct fundamentals | 51.27% | 0.1053% | 5.11% | reject |
| price, direct fundamentals and interactions | 49.01% | 0.0522% | 5.42% | reject |

Direct fundamentals increased average gross return, but did not improve the win
rate and materially worsened the left tail.  This is not an acceptable
accuracy improvement.

## Status

Research-only and shadow-only.  All historical windows have been viewed.  The
study cannot alter rankings, orders, positions, risk controls or execution locks,
and even a historical pass could only justify a separately preregistered fresh
forward study.
