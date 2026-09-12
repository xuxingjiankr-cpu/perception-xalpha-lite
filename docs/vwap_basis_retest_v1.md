# VWAP input-basis correction: reject-only paired re-test

This is a new research version, not recertification. Old reports, source weights,
configs and factor formulas are not changed. No trading, probability-model,
forward-ledger or vendor-pipeline changes. No 456-factor sweep or weight fitting.

## Narrow correction

`build_factor_inputs` now preserves a supplied `panel['vwap']` and only fills
missing values from amount/volume (then the legacy close fallback). The selected
source counts and basis version are stamped in DataFrame attributes and factor
audits. Fallbacks are explicitly **not** certified as adjusted. The research
runner checks corrected VWAP against same-basis daily low/high and fails rather
than accepting an unexplained scale mismatch. An OHLC4 proxy stays a proxy, never
relabelled true transaction VWAP. The upstream panel loader may itself fill
missing source values; preserved-panel provenance is not an archived trade tape.

`legacy_amount_volume_v1` is an explicit diagnostic path reproducing the old
amount/volume + close calculation. `archive_vwap_v2` is the corrected default.
Only `alpha101/alpha_094` (weight 0.0730773681113405) of twelve frozen factors
uses VWAP. The runner asserts exact equality of the other eleven rank matrices.
The existing alpha094 multiplication/exponent discrepancy is **not** changed.

Rank-cache keys now include the input builder code AND the chosen price basis;
old cached ranks cannot be reused accidentally. Panel-cache contents and original
research outputs are not overwritten by this re-test.

## Same-support comparison

Both versions use the SAME in-memory panel, axes and ORIGINAL eligibility, with
no complete-case deletion or reranking. The original available-factor arithmetic
and all existing rank values are preserved. The runner asserts that both
composites and RV20 cover every original eligible cell, failing rather than
shrinking the comparison set. The eligibility mask, per-date counts and full
content hash are written. Compare new-run OLD against new-run CORRECTED.

The initial `vwap_basis_retest_v1.json` complete-case proposal was aborted before
any horizon result: it would have removed 1,412,528 cells and changed all ranks.
It is retained as an audit record, not used for a conclusion. The active config
is `vwap_basis_retest_v2.json`; no evaluation threshold or weight was changed.

The two existing study `run()` entry points consume this paired input explicitly,
using their unchanged splits, horizons, frozen weights, execution eligibility,
resolved-outcome arithmetic and benchmarks restricted to the common mask. Each
gets a separate new output directory. Their original softer verdicts are retained
only as diagnostic fields; the new paired report applies the owner's stricter
criteria. Existing next-session eligibility and overlapping-label t limitations
remain explicit; this task is not a new execution certification.

## Frozen rejection rules

- #12: a predeclared composite (frozen prior or equal weight, NOT a selected single
  factor) has positive day-neutral excess with day-clustered t >=2 at the SAME
  horizon in BOTH validation and shadow. Report all six horizons regardless of
  outcome. Old/new effect sizes and t statistics remain side by side.
- #14: a predeclared composite exceeds RV20's tail-excess AND has lower return
  excess in the excluded worst decile, in BOTH windows at BOTH h=1 and h=5.
  Better tails purchased by excluding more upside do not overturn the conclusion.
- Rejection of a standing negative conclusion is not acceptance for trading.
  These windows have already been viewed; no tuning or automatic promotion.

One requested cheap diagnostic uses existing `qlib158/vstd60`, unchanged. Its
actual formula is `std(volume,60)/current_volume`: **volume variability**, not
60-session price volatility. Compare it with original `-std(returns,20)` (minimum
10 observations, exactly as in the standing study). No new volatility variant.

## Run / verification

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_vwap_basis_retest_v1.py --run-id <unique_run_id>
```

Output root: `outputs/edge_research/vwap_basis_retest_v1/<run_id>/`. Four study
reports (two bases x two existing scripts), paired report/result, config/commit,
protected-file hashes, panel fingerprint, price-basis audit and common-support
mask. Existing run IDs fail closed. Underlying data/signatures and protected
artifacts are checked again after the paired runs.

T174 pins non-trivial adjustment and 7.678x regression rejection, fallback stamps,
future-prefix invariance, actual twelve-factor computation, eleven-factor
invariance, basis-specific caches, both real study entry points, no overwrite,
the VSTD60 definition and conjunctive conclusion criteria. Full replay invariants
and existing frontier/tail/cache tests must pass before a commit.
