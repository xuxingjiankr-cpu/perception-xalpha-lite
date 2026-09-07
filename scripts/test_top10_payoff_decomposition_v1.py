"""Exercise actual estimator and run entry, not a parallel toy evaluator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_top10_payoff_decomposition_v1 as study


class PayoffTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads(study.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.c["training"].update(trainSessions=40, calibrationSessions=20,
                                 minimumTrainDays=30, minimumCalibrationDays=14,
                                 maximumRowsPerDay=50, minimumCrossSection=20)
        self.rng = np.random.default_rng(100)
        self.dates = pd.bdate_range("2023-01-02", periods=225)
        self.daily = {}
        for i in range(225):
            x = self.rng.uniform(size=(80, 16))
            y = .004 * (x[:, 0] - .5) + self.rng.normal(0, .018, 80)
            self.daily[i] = {"x": x, "y": y, "entry": np.ones(80, dtype=bool),
                             "delay": np.zeros(80), "symbols": np.array([f"SH.{600000 + j}" for j in range(80)]),
                             "controls": {k: x.mean(axis=1) for k in ["guarded16", "frozen16", "equal16"]}}

    def test_config_and_purge_fail_closed(self):
        study.validate(self.c)
        for section, key, value in [("safety", "mayTrade", True), ("training", "purgeSessions", 6),
                                    ("evaluation", "topCount", 9), ("evaluation", "roundTripCost", 0)]:
            c = copy.deepcopy(self.c); c[section][key] = value
            with self.assertRaises(ValueError):
                study.validate(c)
        tr, cal = study.split_indices(100, self.c)
        self.assertLess(tr[-1] + 7, cal[0])
        self.assertLess(cal[-1] + 7, 100)

    def test_estimator_future_mutation_and_payoff_identity(self):
        tr, cal = study.split_indices(100, self.c)
        a, _ = study.fit_models(self.daily, tr, cal, self.c)
        changed = copy.deepcopy(self.daily)
        for i in range(93, 225):
            changed[i]["x"] *= 1000
            changed[i]["y"][:] = -10
        b, _ = study.fit_models(changed, tr, cal, self.c)
        pa, pb = study.predict(a, self.daily[100]["x"]), study.predict(b, self.daily[100]["x"])
        for k in pa:
            np.testing.assert_allclose(pa[k], pb[k])
        np.testing.assert_allclose(pa["raw_payoff"], pa["p"] * pa["gain"] - (1 - pa["p"]) * pa["loss"])
        self.assertTrue(((pa["p"] >= 0) & (pa["p"] <= 1)).all())
        self.assertTrue((pa["gain"] >= 0).all() and (pa["loss"] >= 0).all())

    def test_open_execution_never_uses_later_intraday_ohlcv(self):
        ix = pd.bdate_range("2023-01-02", periods=12)
        panel = {k: pd.DataFrame(v, index=ix, columns=["SH.600000"]) for k, v in
                 {"open": 10., "preclose": 10., "close": 10., "high": 10., "low": 10.,
                  "volume": 100., "amount": 1000., "trade_status": 1., "is_st": 0., "membership": True}.items()}
        panel["open"].iloc[2, 0] = 10.1
        a, entry, delay = study.execution_labels(panel, self.c)
        self.assertAlmostEqual(a.iloc[0, 0], .01)
        self.assertEqual(delay.iloc[0, 0], 0)
        b = copy.deepcopy(panel)
        for k in ["close", "high", "low", "volume", "amount"]:
            b[k].iloc[1:] *= 10
        pd.testing.assert_frame_equal(a, study.execution_labels(b, self.c)[0])
        b["open"].iloc[1, 0] = 11.
        self.assertFalse(study.execution_labels(b, self.c)[1].iloc[0, 0])
        b = copy.deepcopy(panel); b["open"].iloc[2, 0] = 9.
        self.assertEqual(study.execution_labels(b, self.c)[2].iloc[0, 0], 1)

    def test_missing_outcomes_never_replace_selected_names(self):
        row = copy.deepcopy(self.daily[100])
        scores = row["controls"]["equal16"]
        selected = study.top10(scores)
        tr, cal = study.split_indices(100, self.c)
        m, _ = study.fit_models(self.daily, tr, cal, self.c)
        pr = study.predict(m, row["x"])
        row["y"][selected[:3]] = np.nan
        row["entry"][selected[:2]] = False
        row["delay"][selected[:3]] = np.nan
        records = study.record_picks(row, 100, self.dates[100], 200, "shadow", "equal16", scores, pr, self.c)
        self.assertEqual([r["securityId"] for r in records], row["symbols"][selected].tolist())
        self.assertEqual(sum(r["state"] == "resolved" for r in records), 7)
        self.assertEqual(sum(r["state"] == "entered_unresolved_exit" for r in records), 1)
        self.assertEqual(sum(r["state"] == "unfilled_entry" for r in records), 2)

    def test_period_end_censors_even_known_future_returns(self):
        row = self.daily[100]
        self.assertTrue(np.isnan(study.period_outcomes(row, 100, 101, self.c)).all())
        np.testing.assert_allclose(study.period_outcomes(row, 100, 102, self.c), row["y"])

    def test_real_run_entry_produces_six_books_and_hindsight_separately(self):
        partitions = {"audit": self.dates[100:135], "validation": self.dates[135:180], "shadow": self.dates[180:]}
        fixture = (self.dates, partitions, self.daily, {"panel": {"limitations": ["synthetic_fixture"]}})
        with tempfile.TemporaryDirectory(prefix="payoff_test_", dir=study.ROOT / "outputs/edge_research") as td:
            c = copy.deepcopy(self.c); c["outputRoot"] = str(Path(td).relative_to(study.ROOT))
            path = Path(td) / "config.json"; path.write_text(json.dumps(c), encoding="utf-8")
            with patch.object(study, "build_data", return_value=fixture):
                result = study.run(path, "entry")
            self.assertFalse(result["mayPromote"])
            self.assertEqual(result["orders"], [])
            self.assertEqual(len(result["summary"]["shadow"]), 6)
            self.assertEqual(len(result["hindsightExposure"]["shadow"]), 3)
            picks = pd.read_csv(Path(td) / "entry/daily_picks.csv")
            self.assertTrue(picks.groupby(["date", "policy"]).size().eq(10).all())
            self.assertEqual(json.loads((Path(td) / "entry/manifest.json").read_text())["status"], "completed")


if __name__ == "__main__":
    unittest.main()
