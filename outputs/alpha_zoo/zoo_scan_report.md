# Alpha-Zoo Daily Scan -- full ETF universe, implementation-lagged, train-ranked/test-reported

panel 388 codes x 1602 days | train<= 2024-12-31 < test | cost 0.1550% RT x turnover | factors scanned 456, usable 454

Top-25 by |TRAIN IC-IR| (TEST columns are out-of-sample for this ranking):

| factor | train IC t | train ICIR | TEST IC t | TEST ICIR | TEST L-S t | TEST long-net IR |
|---|--:|--:|--:|--:|--:|--:|
| alpha101/alpha_011 | 4.19 | 1.862 | 1.83 | 1.507 | 0.95 | -0.34 |
| qlib158/cord60 | -3.82 | -1.74 | -2.4 | -1.973 | 1.54 | 0.423 |
| academic/high52w | 3.39 | 1.684 | 2.36 | 1.94 | -0.16 | -1.042 |
| gtja191/alpha_180 | 3.53 | 1.61 | 1.33 | 1.098 | 0.64 | -0.919 |
| academic/retskew | 3.48 | 1.582 | 1.41 | 1.159 | 0.47 | 0.406 |
| gtja191/alpha_153 | 3.52 | 1.578 | 1.37 | 1.13 | 0.7 | 0.264 |
| gtja191/alpha_010 | 3.41 | 1.531 | 1.58 | 1.301 | 0.84 | 0.25 |
| alpha101/alpha_005 | 3.43 | 1.528 | 1.4 | 1.15 | 0.71 | 0.211 |
| alpha101/alpha_036 | 3.14 | 1.522 | 0.95 | 0.785 | -0.56 | -3.064 |
| qlib158/cord30 | -3.37 | -1.514 | -1.2 | -0.987 | 1.71 | -0.032 |
| alpha101/alpha_041 | -3.33 | -1.478 | -1.42 | -1.17 | -0.66 | -1.186 |
| gtja191/alpha_173 | 3.28 | 1.454 | 1.35 | 1.108 | 0.62 | 0.186 |
| gtja191/alpha_082 | -3.27 | -1.453 | -3.35 | -2.757 | -0.94 | -1.773 |
| gtja191/alpha_126 | 3.18 | 1.412 | 1.4 | 1.151 | 0.65 | 0.177 |
| academic/carhart_mom | 2.79 | 1.39 | 1.67 | 1.375 | 1.26 | 0.127 |
| academic/rmw | 2.98 | 1.357 | -0.28 | -0.228 | -0.62 | -0.025 |
| gtja191/alpha_072 | -3.01 | -1.338 | -3.25 | -2.677 | -1.0 | -1.807 |
| qlib158/max60 | -2.93 | -1.335 | -1.85 | -1.52 | 0.29 | -0.778 |
| qlib158/cntp30 | 2.91 | 1.308 | 2.46 | 2.026 | 1.0 | 0.956 |
| gtja191/alpha_060 | 2.89 | 1.294 | 1.15 | 0.947 | 0.35 | -1.084 |
| gtja191/alpha_140 | 2.68 | 1.284 | -1.09 | -0.898 | -2.03 | -2.823 |
| qlib158/cntd30 | 2.8 | 1.255 | 2.52 | 2.072 | 0.74 | 0.726 |
| qlib158/cntd60 | 2.75 | 1.248 | 2.28 | 1.88 | 0.3 | -0.95 |
| alpha101/alpha_065 | -2.7 | -1.234 | -1.81 | -1.493 | -0.72 | -2.187 |
| academic/hml | -2.46 | -1.225 | -1.84 | -1.518 | -1.31 | -2.254 |

- factors with |TEST IC t| >= 2: 116 / 454 (chance at 5%: ~23)
- **zoo-level PBO** (L-S daily matrix): 0.3571
- **DSR note (best test long-net among train-top25, n_trials=454)**: {'n_trials': 454, 'observed_sharpe': 0.956, 'expected_max_noise_sharpe': 3.333, 'flag': 'consistent_with_luck'}

## Read

A factor graduates ONLY if: TEST IC |t|>=2 AND TEST long-only net IR meaningfully >0 AND it survives the zoo-level multiple-testing discount (DSR) -- then replay integration + forward shadow, per the standing gates. Diagnostic only; no live change from this scan.

## Second-stage checks (2026-07-04, post-hoc candidates cntp30/cntd30)

Candidates picked AFTER seeing test long-net (selection-contaminated; checks below are
robustness reads, not fresh OOS):

1. **Weekly vs daily rebalance**: turnover drops ~3x (0.216 -> 0.077/day) but net IR is
   unchanged (0.956 -> 0.950). Cost is NOT the binding constraint at these turnovers -- the
   long-only spread itself is small (TEST t=1.16 over 18 months).
2. **Year-by-year net spread (cntp30)**: 2020 -5.4%, 2021 +4.2%, 2022 +1.0%, 2023 +1.1%,
   2024 -10.4%, 2025 +2.6%, **2026H1 +32.7% (IR 2.09)** -- the entire apparent edge is one
   half-year regime. cntd30 identical pattern. Not a durable factor; no graduation.

## Final verdict

The zoo carries real cross-sectional INFORMATION on the ETF universe (116/454 factors with
|TEST IC t|>=2 vs ~23 by chance) but NO factor converts it into a long-only, cost-surviving,
regime-stable portfolio edge on 6.5 years of daily data. Consistent with every prior line:
information exists, harvestable edge does not -- the binding constraints are the long-only
restriction, the thin cross-sectional dispersion of ETFs (vs single stocks these factors were
designed for), and regime concentration. No live change; the vendored zoo + scan pipeline stay
for future re-runs (e.g. if the universe or cost structure changes).
