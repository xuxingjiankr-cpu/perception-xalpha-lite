"""Adversarial causal tests for the opt-in research loader."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_ashare_pit_panel_v2 as loader


class PitPanelTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=100)
        self.spec = {"liquidityWindow": 10, "minimumLiquidityObservations": 8,
                     "minimumPriorObservations": 20, "minimumAmountCny": 100,
                     "maximumMissingFraction": .2, "maximumSuspensionFraction": .2,
                     "minimumCrossSection": 1}

    def rows(self):
        return [{"dt": str(d.date()), "open": 10, "high": 11, "low": 9,
                 "close": 10, "vol": 100, "amount": 1000, "preclose": 10,
                 "source": loader.SOURCE, "adjustment": "backward_adjusted_baostock_test",
                 "isST": 0, "tradeStatus": 1, "pointInTimeStatus": True}
                for d in self.dates]

    def build(self, root, rows):
        (root / "bars").mkdir(exist_ok=True)
        (root / "master.jsonl").write_text(json.dumps({
            "securityId": "SH.600000", "exchange": "SH", "stockCode": "600000",
            "listingDate": "2020-01-01", "pointInTimeMembership": True}), encoding="utf-8")
        (root / "bars/SH_600000.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        u = {"masterPath": "master.jsonl", "barsRoot": "bars", "exchanges": ["SH"]}
        with patch.object(loader, "ROOT", root):
            return loader.build_panel(u, self.spec, self.dates)

    def test_future_bad_data_cannot_remove_earlier_symbol(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = self.rows()
            a, _ = self.build(root, rows)
            for r in rows[60:]:
                r.update(amount=0, vol=0, tradeStatus=0, isST=1,
                         source="unknown", adjustment="unadjusted")
            b, audit = self.build(root, rows)
            pd.testing.assert_frame_equal(a["eligible"].iloc[:60], b["eligible"].iloc[:60])
            self.assertEqual(a["close"].columns.tolist(), b["close"].columns.tolist())
            self.assertEqual(audit["maskedUnverifiedSourceOrStatusRows"], 40)
            self.assertFalse(audit["unbiasedHistoricalValidationEligible"])

    def test_history_seasoning_and_current_amount_not_forward_looking(self):
        with tempfile.TemporaryDirectory() as td:
            a, _ = self.build(Path(td), self.rows())
        self.assertFalse(a["eligible"].iloc[:20].any().any())
        self.assertTrue(a["eligible"].iloc[20:60].all().all())
        b = copy.deepcopy(a)
        b["amount"].iloc[40:] = 1
        changed = loader.eligibility(b, self.spec)
        pd.testing.assert_frame_equal(a["eligible"].iloc[:41], changed.iloc[:41])
        self.assertFalse(changed.iloc[-1, 0])

    def test_short_history_and_missing_future_status_do_not_delete_past(self):
        with tempfile.TemporaryDirectory() as td:
            a, _ = self.build(Path(td), self.rows()[:45])
            rows = self.rows()
            for r in rows[45:]:
                r.pop("isST")
            b, _ = self.build(Path(td), rows)
        pd.testing.assert_frame_equal(a["eligible"].iloc[:45], b["eligible"].iloc[:45])
        self.assertFalse(b["eligible"].iloc[45:].any().any())

    def test_forward_filled_status_and_unadjusted_rows_rejected_locally(self):
        rows = self.rows()
        rows[40]["pointInTimeStatusSource"] = "last_status_carried_forward"
        rows[41]["adjustment"] = "unadjusted"
        with tempfile.TemporaryDirectory() as td:
            panel, _ = self.build(Path(td), rows)
        self.assertTrue(np.isnan(panel["close"].iloc[40:42]).all().all())
        self.assertTrue(panel["close"].iloc[39].notna().all())


if __name__ == "__main__":
    unittest.main()
