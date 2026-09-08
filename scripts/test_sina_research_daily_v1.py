"""Offline causal/contract tests, including the real collection entry point."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import collect_sina_research_daily_v1 as c


CFG = {
    "schemaVersion": "sina_daily_research_v1", "researchOnly": True,
    "startDate": "2020-01-02", "endDate": "2020-01-06",
    "dataRoot": "data/market/ashare_research/sina_daily_v1",
    "outputRoot": "outputs/edge_research/sina_daily_v1",
    "masterPaths": ["master.jsonl"], "requestIntervalSeconds": 2,
    "connectTimeoutSeconds": 5, "readTimeoutSeconds": 20,
    "maximumConsecutiveTransportFailures": 5,
    "rawVwapRelativeTolerance": .00001, "rawVwapAbsoluteToleranceCny": .0001,
    "adjustment": "same_vendor_hfq_factor_effective_date_asof_no_future_factor",
    "minimumFreshPriceSymbolsForReview": 3000,
    "mayTrainAutomatically": False, "mayPublishDashboard": False, "mayTrade": False,
}
SESSIONS = ["2020-01-02", "2020-01-03", "2020-01-06"]


def bar(dt="2020-01-02", **kwargs):
    return {"date": dt + "T00:00:00.000Z", "open": 10, "high": 11,
            "low": 9, "close": 10.5, "volume": 100, "amount": 1030, **kwargs}


class SinaDailyTests(unittest.TestCase):
    def test_assignment_rejects_code_and_wrong_symbol(self):
        self.assertEqual(c.parse_assignment('var x="abc"; /* checked */', "x"), "abc")
        for text in ('var x="abc";alert(1)', 'var other="abc";', 'var x=__import__("os")'):
            with self.assertRaises(ValueError):
                c.parse_assignment(text, "x")
        with self.assertRaises(ValueError):
            c.parse_assignment('var x={"a":1,"a":2}', "x")

    def test_decoder_receives_literal_not_remote_program(self):
        decoder = Mock()
        decoder.call.return_value = [bar()]
        self.assertEqual(c.decode_bars('var KLC_K2_sh600000="K2/abc";', "sh600000", decoder), [bar()])
        decoder.call.assert_called_once_with("d", "K2/abc", timeout=10000)

    def test_factors_are_strict(self):
        self.assertEqual(c.parse_factors('var sh600000hfq={"total":1,"data":[{"d":"1900-01-01","f":"1"}]};', "sh600000"), [("1900-01-01", 1)])
        for factor in ('NaN', '-2', '0'):
            with self.assertRaises(ValueError):
                c.parse_factors('var sh600000hfq=' + json.dumps({"total": 1, "data": [{"d": "1900-01-01", "f": factor}]}), "sh600000")

    def test_future_factors_and_rows_cannot_change_earlier_inputs(self):
        rows = [bar(dt) for dt in SESSIONS]
        a = c.normalize(rows, [("1900-01-01", 1), ("2020-01-06", 2)], "SH.600000", SESSIONS, CFG)[1]
        b = c.normalize(rows + [bar("2020-01-07", close=999)],
                        [("1900-01-01", 1), ("2020-01-06", 2), ("2020-01-07", 50)],
                        "SH.600000", SESSIONS, CFG)[1]
        self.assertEqual(a, b)
        self.assertEqual([r["close"] for r in a], [10.5, 10.5, 21])
        self.assertEqual(a[-1]["vwap"], 20.6)

    def test_factor_missing_does_not_fill_backward(self):
        raw, adj, report = c.normalize([bar()], [("2020-01-03", 2)], "SH.600000", SESSIONS, CFG)
        self.assertTrue(raw[0]["priceBasisValid"])
        self.assertIsNone(adj[0]["vwap"])
        self.assertIsNone(adj[0]["close"])
        self.assertEqual(report["rejectedReasons"], {"missing_date_local_factor": 1})

    def test_wrong_volume_units_do_not_become_vwap(self):
        _, adj, report = c.normalize([bar(volume=1)], [("1900-01-01", 1)], "SH.600000", SESSIONS, CFG)
        self.assertFalse(adj[0]["priceBasisValid"])
        self.assertIsNone(adj[0]["vwap"])
        self.assertIn("transaction_vwap_outside_raw_range", report["rejectedReasons"])

    def test_dates_duplicate_and_timezone_fail_closed(self):
        for rows in ([bar(), bar()], [bar(date="2020-01-02T08:00:00.000Z")]):
            with self.assertRaises(ValueError):
                c.normalize(rows, [("1900-01-01", 1)], "SH.600000", SESSIONS, CFG)

    def test_missing_bar_and_unknown_status_not_filled(self):
        rows = [bar(SESSIONS[0], isST=0, tradeStatus=1, future_return=10), bar(SESSIONS[-1])]
        _, adj, report = c.normalize(rows, [("1900-01-01", 1)], "SH.600000", SESSIONS, CFG)
        self.assertEqual(len(adj), 2)
        self.assertEqual(report["missingBarsBetweenObservedEndpoints"], 1)
        self.assertIsNone(adj[0]["isST"])
        self.assertIsNone(adj[0]["tradeStatus"])
        self.assertNotIn("future_return", adj[0])
        self.assertFalse(adj[0]["historicalVintageVerified"])

    def test_invalid_bar_stays_masked(self):
        for row in (bar(volume=0), bar(amount=float("nan")), bar(high=8)):
            _, adj, _ = c.normalize([row], [("1900-01-01", 1)], "SH.600000", SESSIONS, CFG)
            self.assertIsNone(adj[0]["open"])
            json.dumps(adj, allow_nan=False)

    def test_cache_integrity_and_single_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x").write_text("original", encoding="utf-8")
            record = {"files": {"x": c.sha(b"original")}}
            c.verify_cached(record, root)
            (root / "x").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                c.verify_cached(record, root)
            with c.single_instance(root / "lock"):
                with self.assertRaises(ValueError):
                    with c.single_instance(root / "lock"):
                        pass

    def test_access_denial_no_retry(self):
        client = c.PublicClient(CFG)
        client.session.get = Mock(return_value=Mock(status_code=429))
        with self.assertRaises(c.AccessDenied):
            client.get("https://finance.sina.com.cn/test")
        self.assertEqual(client.session.get.call_count, 1)
        client.session.close()

    def test_actual_entry_writes_and_resumes_independent_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts/collect_ashare_research_daily.py").write_text("test helper provenance", encoding="utf-8")
            (root / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
            master = {"securityId": "SH.600000", "pointInTimeMembership": True, "listingDate": "1999-11-10"}
            (root / "master.jsonl").write_text(json.dumps(master) + "\n", encoding="utf-8")
            def git(command, **kwargs):
                return "test_commit\n" if command[1] == "rev-parse" else ""
            client = Mock()
            client.get.side_effect = ['var KLC_K2_sh600000="unused";',
                                     'var sh600000hfq={"total":1,"data":[{"d":"1900-01-01","f":"2"}]};']
            args = argparse.Namespace(config="config.json", run_id="offline_entry_test", resume=False, symbols=None)
            with patch.object(c, "ROOT", root), patch.object(c.subprocess, "check_output", side_effect=git), \
                 patch.object(c, "PublicClient", return_value=client), \
                 patch.object(c, "decode_bars", return_value=[bar(dt) for dt in SESSIONS]), patch("builtins.print"):
                self.assertEqual(c.run(args), 0)
                result_path = root / CFG["outputRoot"] / args.run_id / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual(result["successfulSymbols"], 1)
                self.assertEqual(result["validAdjustedRows"], 3)
                self.assertEqual(result["orders"], [])
                self.assertFalse(result["readyForTraining"])
                args.resume = True
                self.assertEqual(c.run(args), 0)
                self.assertEqual(client.get.call_count, 2)  # No duplicate network download.
                status_path = result_path.with_name("status.json")
                saved_status = status_path.read_text(encoding="utf-8")
                status_path.write_text(json.dumps({"failClosedReason": "AccessDenied:provider_access_denied_http_429"}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "access_denied_run_cannot_resume"):
                    c.run(args)
                self.assertEqual(client.get.call_count, 2)
                status_path.write_text(saved_status, encoding="utf-8")
                altered = copy.deepcopy(CFG)
                altered["rawVwapAbsoluteToleranceCny"] = .5
                (root / "config.json").write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "resume_contract_mismatch"):
                    c.run(args)


if __name__ == "__main__":
    unittest.main()
