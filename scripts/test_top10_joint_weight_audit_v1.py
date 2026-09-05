"""Causal and real-entry tests for the isolated sixteen-factor weight experiment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_top10_joint_weight_audit_v1 as study


class JointWeightTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads(study.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.prior = np.full(16, 1 / 16)
        self.rng = np.random.default_rng(16)
        self.x = self.rng.uniform(size=(120, 16))
        self.y = self.rng.normal(0, .02, 120)

    def test_config_safety_and_maturity(self):
        study.validate(self.c)
        for section, key, value in [("safety", "mayTrade", True),
                                    ("training", "purgeSessions", 6),
                                    ("evaluation", "roundTripCost", 0)]:
            c = copy.deepcopy(self.c)
            c[section][key] = value
            with self.assertRaises(ValueError):
                study.validate(c)

    def test_future_outcomes_cannot_change_prior_weights(self):
        m = study.day_statistics(self.x, self.y, self.prior, self.c)
        moments = [m] * 350
        tr = study.training_indices(300, self.c)
        self.assertLess(tr[-1] + 7, 300)
        spec = self.c["candidates"]["joint_balanced"]
        w = study.fit(moments, tr, self.prior, spec, self.c)
        changed = moments.copy()
        for i in range(294, len(changed)):
            changed[i] = (np.eye(16) * 1e9, np.ones((16, 3)) * 1e9)
        np.testing.assert_allclose(w, study.fit(changed, tr, self.prior, spec, self.c))
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w >= .01 - 1e-8).all() and (w <= .25 + 1e-8).all())

    def test_future_mask_never_backfills_top10(self):
        picks = study.choose(self.x, self.prior)
        y = self.y.copy()
        y[picks[:4]] = np.nan
        row = study.basket_row("d", "p", "n", self.x, self.prior, y, self.c)
        self.assertEqual(row["selected"], 10)
        self.assertEqual(row["resolved"], 6)
        self.assertEqual(row["missing"], 4)
        self.assertAlmostEqual(row["net"], self.y[picks[4:]].mean() - .003)
        all_missing = study.basket_row("d", "p", "n", self.x, self.prior, y * np.nan, self.c)
        self.assertTrue(np.isnan(all_missing["up"]))
        self.assertTrue(np.isnan(all_missing["net"]))

    def test_volatility_residual_is_cross_sectional(self):
        v = self.rng.uniform(size=120)
        residual = study.transform(self.x, v, True)
        np.testing.assert_allclose((v - v.mean()) @ residual, 0, atol=1e-12)
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=1e-12)

    def test_return_task_is_market_neutral_in_training(self):
        y = self.rng.uniform(.005, .01, 120)
        a = study.day_statistics(self.x, y, self.prior, self.c)
        b = study.day_statistics(self.x, y + .005, self.prior, self.c)
        np.testing.assert_allclose(a[1][:, 0], b[1][:, 0], atol=1e-12)

    def test_real_run_entry_writes_manifest_metrics_and_all_six_weights(self):
        dates = pd.bdate_range("2024-01-01", periods=200)
        columns = [f"{i:06d}.SZ" for i in range(120)]
        shape = (len(dates), len(columns))
        features = {f"f{i}": pd.DataFrame(self.rng.uniform(size=shape), index=dates, columns=columns) for i in range(16)}
        panel = {"close": pd.DataFrame(10., index=dates, columns=columns)}
        support = panel["close"].notna()
        vol = features["f0"]
        y = pd.DataFrame(self.rng.normal(.001, .02, shape), index=dates, columns=columns)
        delay = y * 0
        adaptive = pd.DataFrame(1 / 16, index=dates, columns=list(features))
        partitions = {"audit": dates[60:100], "validation": dates[100:150], "shadow": dates[150:]}
        fixture = (panel, features, support, vol, y, support, delay, self.prior, adaptive, partitions, {})
        c = copy.deepcopy(self.c)
        c["training"].update(lookbackSessions=40, minimumDays=30)
        root = study.ROOT / "outputs/edge_research"
        with tempfile.TemporaryDirectory(prefix="joint_weight_test_", dir=root) as td:
            c["outputRoot"] = str(Path(td).relative_to(study.ROOT))
            p = Path(td) / "fixture.json"
            p.write_text(json.dumps(c), encoding="utf-8")
            with patch.object(study, "build_data", return_value=fixture):
                result = study.run(p, "entry_test")
            self.assertEqual(result["orders"], [])
            self.assertFalse(result["mayPromote"])
            self.assertEqual(len(result["latestWeights"]), 6)
            self.assertEqual(len(result["summary"]["shadow"]), 9)
            self.assertEqual(result["rejectOnlySurvivors"], [])
            manifest = json.loads((Path(td) / "entry_test/manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            frame = pd.read_csv(Path(td) / "entry_test/daily_baskets.csv")
            self.assertTrue((frame.selected == 10).all())


if __name__ == "__main__":
    unittest.main()
