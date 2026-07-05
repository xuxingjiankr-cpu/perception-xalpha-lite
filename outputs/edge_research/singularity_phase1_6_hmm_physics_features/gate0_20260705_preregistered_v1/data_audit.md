# Singularity Phase 1.6 Gate 0 data audit

- Status: `exploratory_gate0_only`
- Range: 2024-06-12 through 2026-07-03
- Trading days / symbols / bars: 500 / 18 / 425,760
- Break date: `break_date_not_preregistered`
- LPPLS coverage: 100.00%
- DMD coverage: 60.00%
- Gate 0 passed: `False`
- Formal modeling allowed: `False`

Observed limitations: raw bars have no adjustment factor; the timezone is inferred; bid/ask spreads and source timestamps are absent; the fixed current-product universe creates survivorship bias; DMD completeness excludes early-session decisions.

No HMM was fitted because the break date is not preregistered.
