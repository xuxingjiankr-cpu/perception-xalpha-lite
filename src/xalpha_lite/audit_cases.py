"""Three synthetic audit demonstrations. No vendor, broker, or research ledger access."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .discovery import deflated_sharpe_ratio, pbo
from .pit import align_point_in_time_fundamentals

SEED = 20260914
SCHEMA = "xalpha_synthetic_audit_cases_v1"


def _rounded(value: float) -> float:
    return round(float(value), 6)


def noise_case() -> tuple[dict, dict[str, pd.DataFrame]]:
    """One fixed draw; never search seeds until a diagnostic looks attractive."""
    rng = np.random.default_rng(SEED)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (756, 64)),
        index=pd.bdate_range("2020-01-01", periods=756, name="date"),
        columns=[f"noise_{i:02d}" for i in range(64)],
    )
    train, test = returns.iloc[:504], returns.iloc[504:]
    sharpes = returns.mean() / returns.std(ddof=1)
    selected = (train.mean() / train.std(ddof=1)).idxmax()
    hindsight = (test.mean() / test.std(ddof=1)).idxmax()
    dsr = deflated_sharpe_ratio(returns[sharpes.idxmax()], sharpes.tolist(), 64)
    cscv = pbo(returns, blocks=8)
    return {
        "id": "noise_selection",
        "title": "A high Sharpe can come from selecting noise",
        "data_kind": "synthetic_iid_zero_mean_returns",
        "seed": SEED,
        "observations": 756,
        "variants": 64,
        "declared_trials": 64,
        "train_rows": 504,
        "evaluation_rows": 252,
        "annualization_sessions": 244,
        "full_sample_winner": str(sharpes.idxmax()),
        "best_sharpe_ann": dsr["observed_sharpe_ann"],
        "expected_max_sharpe_ann": dsr["expected_max_trial_sharpe_ann"],
        "dsr_statistic": dsr["probability"],
        "pbo": cscv["pbo"],
        "cscv_splits": cscv["splits"],
        "train_selected_variant": str(selected),
        "hindsight_selected_variant": str(hindsight),
        "hindsight_evaluation_bps": _rounded(test[hindsight].mean() * 1e4),
        "train_selected_evaluation_bps": _rounded(test[selected].mean() * 1e4),
        "same_evaluation_support": True,
        "interpretation": "Full-sample selection is a deliberate negative control. "
        "The two selections are evaluated on the identical last 252 rows. "
        "The data-generating process has zero expected return; a realized gain is not alpha.",
        "limitations": "DSR is an approximate evidence statistic, not a posterior probability "
        "of profitability. CSCV is not chronological forward validation. No costs, execution "
        "or real-market structure are modeled in this IID teaching fixture.",
    }, {"noise_returns.csv": returns}


def disclosure_case() -> tuple[dict, dict[str, pd.DataFrame]]:
    sessions = pd.bdate_range("2020-01-01", "2020-03-10", name="date")
    statements = pd.DataFrame([
        ["TOY_A", "2019-12-31", "2020-01-03", "2020-01-03", 1.0],
        ["TOY_A", "2020-01-31", "2020-02-14", "2020-02-14", 4.0],
        ["TOY_A", "2020-01-31", "2020-02-14", "2020-03-03", 2.5],
    ], columns=["symbol", "report_date", "notice_date", "update_date", "eps"])
    safe = align_point_in_time_fundamentals(statements, sessions)["eps"]["TOY_A"]
    # Deliberately WRONG control: backdate the latest restatement to the report period end.
    latest = statements.drop_duplicates(["symbol", "report_date"], keep="last")
    wrong = pd.Series(latest.eps.to_numpy(), index=pd.to_datetime(latest.report_date))
    wrong = wrong.reindex(wrong.index.union(sessions)).sort_index().ffill().reindex(sessions)
    mismatch = wrong.notna() & (safe.isna() | wrong.ne(safe))
    comparison = pd.DataFrame({"report_date_alignment_wrong": wrong,
                               "disclosure_alignment": safe, "unsafe_row": mismatch})
    first = safe[safe.eq(4.0)].index.min()
    restated = safe[safe.eq(2.5)].index.min()
    return {
        "id": "disclosure_timing", "title": "A reporting period is not an availability date",
        "data_kind": "synthetic_disclosures_and_weekday_calendar",
        "calendar": "Synthetic weekdays, NOT an exchange holiday calendar",
        "rows": len(comparison), "mismatched_rows": int(mismatch.sum()),
        "first_disclosed_value_available": first.strftime("%Y-%m-%d"),
        "restatement_available": restated.strftime("%Y-%m-%d"),
        "bad_value_on_2020_02_03": float(wrong.loc["2020-02-03"]),
        "safe_value_on_2020_02_03": float(safe.loc["2020-02-03"]),
        "interpretation": "The existing PIT aligner exposes a value only on the next session "
        "strictly after max(notice_date, update_date). A later restatement must not alter "
        "earlier features. Missing notice_date raises ValueError.",
        "limitations": "The example audits availability, not return prediction. Real studies "
        "also need exchange calendars, timestamps and historical vendor vintages; an aligner "
        "cannot reconstruct revisions that a vendor discarded.",
    }, {"disclosures.csv": statements, "pit_alignment.csv": comparison}


def select_vwap_for_case(frame: pd.DataFrame) -> pd.Series:
    """A teaching adapter, not a new production loader or a vendor certification."""
    if "archive_vwap" in frame:
        selected = frame["archive_vwap"].copy()
        basis = frame["archive_vwap_basis"]
    else:
        selected = frame["amount"].div(frame["volume"])
        basis = frame["cash_price_basis"]
    if not basis.eq(frame["close_basis"]).all():
        raise ValueError("mixed_price_basis: VWAP and close use different normalizations")
    valid = np.isfinite(selected) & selected.gt(0) & selected.between(frame["low"], frame["high"])
    if not valid.all():
        raise ValueError("invalid_vwap: missing, nonpositive, or outside this fixture's OHLC")
    return selected


def price_basis_case() -> tuple[dict, dict[str, pd.DataFrame]]:
    raw_close = np.array([100., 110., 80., 50.])
    adjustment = np.array([2., 1., 5., 3.])
    frame = pd.DataFrame({
        "symbol": ["TOY_A", "TOY_B", "TOY_C", "TOY_D"],
        "raw_close": raw_close, "adjustment_multiplier": adjustment,
        "open": raw_close * adjustment * [1.01, 0.99, 1.005, 1.0],
        "high": raw_close * adjustment * [1.03, 1.04, 1.02, 1.01],
        "low": raw_close * adjustment * [0.99, 0.98, 0.985, 0.975],
        "close": raw_close * adjustment, "volume": [1000., 2000., 1500., 1200.],
        "close_basis": "adjusted", "archive_vwap_basis": "adjusted",
        "cash_price_basis": "raw",
    }).set_index("symbol")
    frame["amount"] = raw_close * np.array([1.005, 1.01, 0.998, 0.995]) * frame.volume
    frame["archive_vwap"] = frame[["open", "high", "low", "close"]].mean(axis=1)
    corrected = select_vwap_for_case(frame)
    wrong = frame.amount.div(frame.volume)
    old_score, corrected_score = wrong.div(frame.close) - 1, corrected.div(frame.close) - 1
    comparison = pd.DataFrame({
        "close": frame.close, "cash_vwap_wrong_basis": wrong,
        "archive_proxy_correct_basis": corrected,
        "wrong_score": old_score, "corrected_score": corrected_score,
        "wrong_rank": old_score.rank(ascending=False),
        "corrected_rank": corrected_score.rank(ascending=False),
    })
    return {
        "id": "price_basis", "title": "Mixing price bases can change a cross-sectional rank",
        "data_kind": "synthetic_adjusted_bars_and_raw_cash_amounts",
        "symbols": frame.index.tolist(), "rows": len(frame), "same_support": True,
        "factor": "vwap / close - 1 (toy expression, not a validated factor)",
        "vwap_source": "adjusted_ohlc4_proxy_not_true_transaction_vwap",
        "max_archive_to_cash_ratio": _rounded(corrected.div(wrong).max()),
        "rank_changes": int(comparison.wrong_rank.ne(comparison.corrected_rank).sum()),
        "wrong_order": old_score.sort_values(ascending=False).index.tolist(),
        "corrected_order": corrected_score.sort_values(ascending=False).index.tolist(),
        "basis_guard": "rejects raw cash fallback against adjusted OHLC",
        "interpretation": "Preserve the supplied adjusted OHLC4 proxy with its provenance. "
        "Raw amount/raw volume is not an adjusted price. Symbol-specific adjustment histories "
        "need not cancel in ranks; no dates or names were removed from the comparison.",
        "limitations": "OHLC4 is not transaction VWAP, and basis consistency is not predictive "
        "power. This explicit-metadata fixture cannot certify arbitrary vendor data. The "
        "adapter is demonstration-only and does not change the discovery input loader.",
    }, {"price_basis_inputs.csv": frame, "price_basis_comparison.csv": comparison}


def build_cases() -> tuple[dict, dict[str, pd.DataFrame]]:
    cases, tables = [], {}
    for build in (noise_case, disclosure_case, price_basis_case):
        case, inputs = build()
        cases.append(case)
        tables.update(inputs)
    return {"schema_version": SCHEMA, "status": "synthetic_demonstration_only",
            "source": "Generated locally; no private data, securities or empirical results",
            "orders": [], "automatic_trading_changes": [], "cases": cases}, tables


def markdown_report(report: dict) -> str:
    noise, pit, basis = report["cases"]
    return f"""# Three synthetic audit cases

Generated by `python examples/run_audit_cases.py`. Synthetic demonstration only.
All data are generated locally. No market forecasts, private results or orders.

| Noise selection | Value |
|---|---:|
| Variants / declared trials | {noise['variants']} / {noise['declared_trials']} |
| Best full-sample annualized Sharpe | {noise['best_sharpe_ann']} |
| Expected maximum Sharpe under the approximation | {noise['expected_max_sharpe_ann']} |
| DSR statistic (not probability of profitability) | {noise['dsr_statistic']} |
| CSCV/PBO | {noise['pbo']} |
| Hindsight selection on final 252 rows, bps/day | {noise['hindsight_evaluation_bps']} |
| Train-only selection on the SAME 252 rows, bps/day | {noise['train_selected_evaluation_bps']} |

## Disclosure timing

Incorrect alignment changes **{pit['mismatched_rows']}** of {pit['rows']} rows.
The new value first appears on **{pit['first_disclosed_value_available']}**;
its restatement first appears on **{pit['restatement_available']}**.
These are synthetic weekdays, not exchange-calendar claims.

## Price basis

Maximum archive-proxy / raw-cash-price ratio: **{basis['max_archive_to_cash_ratio']}x**.
Changed ranks: **{basis['rank_changes']} / {basis['rows']}**, on identical symbols.
Wrong order: {', '.join(basis['wrong_order'])}.
Consistent-basis order: {', '.join(basis['corrected_order'])}.
The preserved field is an **OHLC4 proxy, not transaction VWAP**.

## What these checks do not establish

""" + "\n\n".join(f"- **{case['title']}**: {case['limitations']}" for case in report["cases"]) + "\n"


def write_cases(output: Path) -> dict:
    report, tables = build_cases()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / name, index=name != "disclosures.csv", float_format="%.12g")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    paths = [output / name for name in tables] + [output / "report.json", output / "report.md"]
    manifest = {
        "schema_version": SCHEMA, "data_kind": "synthetic", "seed": SEED,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("discovery.py", "pit.py")
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "orders": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit-cases"))
    args = parser.parse_args()
    report = write_cases(args.output_dir)
    print(markdown_report(report))
    print(f"Artifacts: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
