"""Adversarial contracts plus REAL factor/rank and readiness entry tests."""
from __future__ import annotations

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_top10_strict_inputs_v1 as strict
import research_top10_upgrade_readiness_v1 as readiness


def rows_fixture(n=320):
    dates = pd.bdate_range("2020-01-01", periods=n)
    adjusted, raw = [], []
    for i, dt in enumerate(dates):
        for j in range(4):
            close = 10 + j + .01 * i + .1 * np.sin(.2 * i + j)
            vol = 1000000 + 250000 * np.sin(.07 * i + j) + 150000 * np.cos(.4 * i + .3 * j) + 300 * i
            row = {"securityId": f"SH.{600000 + j}", "dt": str(dt.date()),
                   "source": strict.SOURCE, "adjustflag": "3", "adjustment": "none_raw_baostock",
                   "amountSource": "exchange_reported_via_baostock", "pointInTimeStatus": True,
                   "volumeUnit": "shares", "amountUnit": "CNY", "isST": 0, "tradeStatus": 1,
                   "open": close - .02, "high": close + .08, "low": close - .08, "close": close,
                   "vol": vol, "amount": vol * (close - .015 + .01 * np.sin(.71 * i + j))}
            raw.append(row)
            adj = dict(row, adjustflag="1", adjustment="backward_adjusted_baostock_pctchg_method",
                       vwap=9999999, vwapSource="adjusted_ohlc4_proxy_not_true_transaction_vwap")
            for k in strict.OHLC:
                adj[k] *= 2 + j / 10
            adjusted.append(adj)
    return dates, adjusted, raw


def config():
    return json.loads(readiness.DEFAULT.read_text(encoding="utf-8"))


def picks_fixture(n=28):
    rows = []
    for date in pd.bdate_range("2020-01-01", periods=n):
        for policy in ["base", "new"]:
            for rank in range(1, 11):
                gross = .025 if policy == "new" and rank <= 8 else (-.004 if policy == "new" else .001)
                rows.append({"date": str(date.date()), "period": "test", "policy": policy,
                             "rank": rank, "securityId": f"SH.{600000 + rank}", "state": "resolved",
                             "pUp": .8, "gross": gross, "net": gross - .003})
    return pd.DataFrame(rows)


class StrictUpgradeTests(unittest.TestCase):
    def test_verified_pair_not_proxy_and_explicit_units(self):
        _, a, r = rows_fixture(1)
        expected = r[0]["amount"] / r[0]["vol"] * 2
        self.assertAlmostEqual(strict.paired_vwap(a[0], r[0]), expected)
        for key, value in [("volumeUnit", "hands"), ("amountUnit", "thousand_CNY"),
                           ("source", "mootdx_tdx"), ("pointInTimeStatusSource", "last_status_carried")]:
            bad = dict(r[0], **{key: value})
            with self.assertRaises(strict.ContractError):
                strict.paired_vwap(a[0], bad)

    def test_pair_contradictions_and_no_future_scale(self):
        _, a, r = rows_fixture(2)
        for bad in [None, r[4], dict(r[0], amount=r[0]["amount"] * 2),
                    dict(r[0], high=r[0]["high"] * 2), dict(r[0], isST=1)]:
            with self.assertRaises(strict.ContractError):
                strict.paired_vwap(a[0], bad)
        bad = dict(r[0], amount=r[0]["vol"] * 20)
        with self.assertRaisesRegex(strict.ContractError, "outside_raw_range"):
            strict.paired_vwap(dict(a[0], amount=bad["amount"]), bad)

    def test_missing_pair_null_and_extra_future_fields_not_exposed(self):
        dates, a, r = rows_fixture(3)
        a[0]["future_return"] = 100
        panel, audit = strict.build_factor_inputs(a, r[1:], dates)
        self.assertTrue(np.isnan(panel["vwap"].iloc[0, 0]))
        self.assertNotIn("future_return", panel)
        self.assertEqual(audit["rejected"]["missing_official_raw_companion"], 1)
        panel2, _ = strict.build_factor_inputs(a[4:], r[4:], dates)
        self.assertTrue(panel2["returns"].iloc[1].isna().all())
        a[0]["source"] = "mootdx_raw_normalized_to_baostock_scale"
        panel3, _ = strict.build_factor_inputs(a, r, dates)
        self.assertTrue(panel3["close"].iloc[0, :1].isna().all())

    def test_real_gtja131_rank_entry_complete_book_and_future_causality(self):
        dates, a, r = rows_fixture()
        eligible = pd.DataFrame(True, index=dates, columns=[f"SH.{600000+j}" for j in range(4)])
        factors = [{"factorKey": "gtja191/alpha_070", "direction": -1, "weight": .5},
                   {"factorKey": "gtja191/alpha_131", "direction": 1, "weight": .5}]
        ranks, score, audit = strict.compute_rank_book(a, r, dates, eligible, factors)
        self.assertGreater(score.notna().sum().sum(), 100)
        inputs, _ = strict.build_factor_inputs(a, r, dates)
        m = importlib.import_module("src.factors.zoo.gtja191.alpha_131")
        other = importlib.import_module("src.factors.zoo.gtja191.alpha_070")
        raw = m.compute(inputs)
        history = inputs["vwap"].notna().rolling(252, min_periods=252).sum().eq(252)
        expected = raw.where(history & other.compute(inputs).notna()).rank(axis=1, pct=True)
        pd.testing.assert_frame_equal(ranks["gtja191/alpha_131"], expected)
        self.assertTrue((inputs["vwap"] != inputs["amount"] / inputs["volume"]).all().all())
        mutated_a, mutated_r = copy.deepcopy(a), copy.deepcopy(r)
        for aa, rr in zip(mutated_a[1160:], mutated_r[1160:]):
            for k in strict.OHLC:
                aa[k] *= 3
                rr[k] *= 3
            aa["amount"] *= 3
            rr["amount"] *= 3
        _, mutated, _ = strict.compute_rank_book(mutated_a, mutated_r, dates, eligible, factors)
        pd.testing.assert_frame_equal(score.iloc[:290], mutated.iloc[:290])
        _, missing, _ = strict.compute_rank_book(a, r[:-1], dates, eligible, factors)
        self.assertTrue(np.isnan(missing.iloc[-1, -1]))
        self.assertFalse(audit["mayPromote"])

    def test_duplicate_and_malformed_json_are_fail_closed(self):
        dates, a, r = rows_fixture(1)
        with self.assertRaisesRegex(strict.ContractError, "duplicate_bar"):
            strict.build_factor_inputs(a + [a[0]], r, dates)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SH_600000.jsonl"
            path.write_text(json.dumps(a[0]) + "\n{bad", encoding="utf-8")
            with self.assertRaisesRegex(strict.ContractError, "invalid_jsonl"):
                strict.read_rows(path)

    def test_all_current_twelve_factor_modules_reach_real_compute_entry(self):
        dates, a, r = rows_fixture(320)
        keys = ["gtja191/alpha_070", "gtja191/alpha_052", "qlib158/vstd60", "gtja191/alpha_097",
                "academic/retskew", "gtja191/alpha_145", "qlib158/cord30", "gtja191/alpha_063",
                "alpha101/alpha_029", "alpha101/alpha_094", "alpha101/alpha_088", "qlib158/min5"]
        factors = [{"factorKey": k, "direction": 1, "weight": 1} for k in keys]
        eligible = pd.DataFrame(True, index=dates, columns=[f"SH.{600000+j}" for j in range(4)])
        ranks, score, audit = strict.compute_rank_book(a, r, dates, eligible, factors)
        self.assertEqual(set(ranks), set(keys))
        self.assertEqual(audit["factorCount"], 12)
        self.assertGreater(audit["completeObservations"], 20)
        # The actual twelve-module success path must produce scores, not only NaNs.
        valid = pd.concat(ranks, axis=1).notna().T.groupby(level=1).all().T
        pd.testing.assert_frame_equal(score.notna(), valid.reindex_like(score))
        pd.testing.assert_frame_equal(score, sum(ranks[k] / 12 for k in keys))
        self.assertFalse(audit["internalFactorMissingDataSemanticsCertified"])

    def test_targets_are_not_probability_floor_or_trade_authority(self):
        c = readiness.validate(config())
        result = readiness.assess_top10(picks_fixture(), c, "new", "base", "test")
        self.assertTrue(result["numericalTargetsMet"])
        self.assertAlmostEqual(result["pairedNetLift"], .0182)
        self.assertFalse(result["mayPromote"])
        self.assertFalse(result["dailyGuarantee"])
        self.assertEqual(result["orders"], [])
        bad = picks_fixture()
        bad.loc[(bad.policy == "new") & (bad["rank"] == 10), "pUp"] = .49
        result = readiness.assess_top10(bad, c, "new", "base", "test")
        self.assertFalse(result["numericalTargetsMet"])
        self.assertEqual(result["predictedProbabilityRange"][0], .49)

    def test_unresolved_and_quota_cannot_mechanically_pass(self):
        picks = picks_fixture()
        picks.loc[0, "state"] = "entered_unresolved"
        result = readiness.assess_top10(picks, config(), "new", "base", "test")
        self.assertEqual(result["pairedCompleteDays"], 27)
        self.assertFalse(result["numericalTargetsMet"])
        with self.assertRaisesRegex(ValueError, "ten_preselected"):
            readiness.assess_top10(picks.iloc[1:], config(), "new", "base", "test")
        picks = picks_fixture()
        picks.loc[0, "net"] += .0001
        with self.assertRaisesRegex(ValueError, "cost_basis"):
            readiness.assess_top10(picks, config(), "new", "base", "test")

    def test_contract_mutations_are_rejected(self):
        for key, val in [("pairedMeanNetReturnLift", .001), ("dailyGuarantee", True), ("topCount", 3)]:
            c = config()
            c["targets"][key] = val
            with self.assertRaises(ValueError):
                readiness.validate(c)
        c = config()
        c["safety"]["mayPublishDashboard"] = True
        with self.assertRaises(ValueError):
            readiness.validate(c)

    def test_actual_readiness_entry_audits_files_and_never_trains(self):
        _, a, _ = rows_fixture(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bars = root / "bars"
            bars.mkdir()
            for sid in sorted({row["securityId"] for row in a}):
                (bars / (sid.replace(".", "_") + ".jsonl")).write_text(
                    "\n".join(json.dumps(row) for row in a if row["securityId"] == sid), encoding="utf-8")
            c = config()
            c["priceData"].update(adjustedRoot="bars", rawCompanionRoot="raw_missing")
            cfg = root / "test.json"
            cfg.write_text(json.dumps(c), encoding="utf-8")
            picks = root / "picks.csv"
            picks_fixture().to_csv(picks, index=False)
            with patch.object(readiness, "ROOT", root), patch.object(readiness.subprocess, "check_output", return_value="fixture"):
                result = readiness.run(cfg, "synthetic", picks, "test", "new", "base")
                self.assertEqual(result["exitCode"], 2)
                self.assertFalse(result["newModelTrained"])
                out = root / c["outputRoot"] / "synthetic"
                price = json.loads((out / "price_audit.json").read_text(encoding="utf-8"))
                self.assertEqual(price["rejections"]["missing_official_raw_companion"], 12)
                self.assertEqual(price["legacyRawVwapOutsideAdjustedRangeRows"], 12)
                self.assertTrue(result["existingHistoricalTargetsMet"])
                with self.assertRaises(FileExistsError):
                    readiness.run(cfg, "synthetic")


if __name__ == "__main__":
    unittest.main()
