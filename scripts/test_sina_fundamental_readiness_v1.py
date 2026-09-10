import copy
import unittest
import pandas as pd

from audit_sina_fundamental_readiness_v1 import summarize_fundamentals, trusted_status


class FundamentalReadinessTests(unittest.TestCase):
    def test_date_local_status_rejects_carry_forward(self):
        row = {"securityId": "SH.600000", "source": "baostock_query_history_k_data_plus",
               "pointInTimeStatus": True, "isST": 0, "tradeStatus": 1}
        self.assertTrue(trusted_status(row, "SH.600000"))
        for change in ({"pointInTimeStatusSource": "carried_forward"}, {"isST": None},
                       {"isST": False}, {"source": "mootdx_raw_normalized_to_baostock_scale"}):
            self.assertFalse(trusted_status({**row, **change}, "SH.600000"))

    def test_fundamentals_do_not_use_future_or_fiscal_date(self):
        sessions = pd.bdate_range("2020-01-01", periods=30)
        cfg = {"families": {"quality": {"candidates": [{"id": "roe", "field": "roePct",
               "transform": "level", "scale": 15., "orientation": 1.}]}},
               "fundamentals": {"maximumSignalAgeTradingDays": 130}}
        rows = [{"noticeDate": "2020-01-10", "updateDate": "2020-01-15", "reportDate": "2019-12-31", "roePct": 12.}]
        a = summarize_fundamentals(rows, sessions, "SH.600000", cfg)
        b = summarize_fundamentals(rows + [{**rows[0], "noticeDate": "2025-01-01", "roePct": 99}], sessions, "SH.600000", cfg)
        for key in ("events", "latestEventDate", "latestEventAgeSessions", "latestFamilyComplete"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["latestEventDate"], "2020-01-16")
        missing = copy.deepcopy(rows)
        missing[0]["noticeDate"] = None
        self.assertEqual(summarize_fundamentals(missing, sessions, "SH.600000", cfg)["events"], 0)


if __name__ == "__main__":
    unittest.main()
