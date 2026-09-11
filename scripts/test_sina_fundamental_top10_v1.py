import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_sina_fundamental_top10_v1 as m


class SinaFundamentalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((m.ROOT / 'configs/research/sina_fundamental_top10_v1.json').read_text(encoding='utf-8'))
        self.sid = 'SH.600000'

    def rows(self):
        raw, adj, status = [], [], []
        dates = pd.bdate_range('2020-01-01', periods=12)
        for dt in dates.strftime('%Y-%m-%d'):
            r = {'dt': dt, 'securityId': self.sid, 'source': m.source.SOURCE, 'adjustment': 'none_raw_sina',
                 'volumeUnit': 'shares', 'amountUnit': 'CNY', 'open': 10., 'close': 10., 'high': 11.,
                 'low': 9., 'amount': 10000., 'volume': 1000., 'vwap': 10., 'priceBasisValid': True}
            a = {**r, 'adjustment': 'same_vendor_hfq_factor_effective_date_asof_no_future_factor',
                 'factor': 2., 'factorEffectiveDate': '2019-01-01', 'vwap': 20.,
                 **{k: r[k] * 2 for k in m.source.OHLC}}
            s = {'dt': dt, 'securityId': self.sid, 'source': 'baostock_query_history_k_data_plus',
                 'pointInTimeStatus': True, 'isST': 0, 'tradeStatus': 1}
            raw.append(r); adj.append(a); status.append(s)
        master = {'securityId': self.sid, 'listingDate': '2000-01-01', 'pointInTimeMembership': True}
        return raw, adj, status, master, dates

    def test_same_vendor_arithmetic_and_exact_status_only(self):
        r, a, s, master, dates = self.rows()
        s[4]['pointInTimeStatusSource'] = 'carried_forward'
        s[5]['close'] = 999999  # foreign prices must never be read
        f = m.symbol_frame(r, a, s, master, dates)
        self.assertTrue(pd.isna(f.is_st.iloc[4]))
        self.assertEqual(f.close.iloc[5], 20.)
        a[5]['close'] = 888
        with self.assertRaisesRegex(ValueError, 'adjustment_arithmetic'):
            m.symbol_frame(r, a, s, master, dates)

    def test_unknown_factor_does_not_impute_and_duplicates_reject(self):
        r, a, s, master, dates = self.rows()
        a[2].update(factor=None, factorEffectiveDate=None, priceBasisValid=False)
        f = m.symbol_frame(r, a, s, master, dates)
        self.assertTrue(pd.isna(f.raw_open.iloc[2]))
        a[2].update(factor=2., factorEffectiveDate='2030-01-01', priceBasisValid=True)
        with self.assertRaisesRegex(ValueError, 'future_or_missing'):
            m.symbol_frame(r, a, s, master, dates)
        with self.assertRaisesRegex(ValueError, 'duplicate_date'):
            m.indexed(r + [r[0]], self.sid)

    def test_opening_labels_ignore_future_ohlc_and_keep_t_plus_one(self):
        r, a, s, master, dates = self.rows()
        f = m.symbol_frame(r, a, s, master, dates)
        y, entry, delay = m.labels(f, self.sid, self.c)
        self.assertTrue(entry[0])
        self.assertEqual(delay[0], 0)
        self.assertEqual(y[0], 0)
        q = f.copy()
        q['high'], q['low'], q['close'], q['amount'], q['volume'] = 999., .001, 888., 0., 0.
        for left, right in zip((y, entry, delay), m.labels(q, self.sid, self.c)):
            np.testing.assert_equal(left, right)
        q = f.copy()
        q.loc[dates[2], 'trade_status'] = 0
        self.assertEqual(m.labels(q, self.sid, self.c)[2][0], 1)
        q.loc[dates[3]:, 'factor'] = 3.
        self.assertTrue(np.isnan(m.labels(q, self.sid, self.c)[0][0]))
        q = f.copy()
        q.loc[dates[1], 'raw_open'] = 11.
        self.assertFalse(m.labels(q, self.sid, self.c)[1][0])

    def test_features_are_prefix_invariant_and_missing_disclosure_resets(self):
        idx = pd.bdate_range('2020-01-01', periods=200)
        frame = pd.DataFrame({'close': np.linspace(10, 12, 200), 'volume': np.arange(200) + 10}, index=idx)
        fc = {'families': {k: {'candidates': [{'id': k, 'field': 'value', 'transform': 'level', 'orientation': 1., 'scale': 1.}]} for k in 'abcd'}}
        rows = [{'noticeDate': '2020-01-02', 'reportDate': '2019-12-31', 'value': 1.},
                {'noticeDate': '2020-04-01', 'reportDate': '2020-03-31', 'value': None}]
        old = m.feature_arrays(frame, rows, self.sid, fc, self.c)[0]
        future = frame.copy()
        future.iloc[130:] *= 100
        new_rows = rows + [{'noticeDate': str(idx[140].date()), 'reportDate': '2020-06-30', 'value': 999.}]
        new = m.feature_arrays(future, new_rows, self.sid, fc, self.c)[0]
        np.testing.assert_equal(old[:130], new[:130])
        at = idx.searchsorted('2020-04-01', side='right')
        self.assertTrue(np.isnan(old[at:130, :4]).all())
        self.assertTrue(np.isfinite(old[20, :4]).all())

    def test_common_support_rank_and_purge(self):
        x = np.random.default_rng(1).normal(size=(30, 12))
        arms = m.rank_features(x)
        self.assertEqual([arms[k].shape[1] for k in m.ARMS[1:]], [4, 8, 12, 12])
        np.testing.assert_equal(arms['context_model'], arms['interaction_model'][:, :8])
        x[0, 11] = np.nan
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            m.rank_features(x)
        train, cal = m.payoff.split_indices(400, self.c)
        self.assertLess(max(train) + 7, min(cal))
        self.assertLess(max(cal) + 7, 400)
        broken = copy.deepcopy(self.c)
        broken['training']['purgeSessions'] = 6
        with self.assertRaisesRegex(ValueError, 'purge'):
            m.validate(broken)

    def test_scalar_price_validation_is_numerically_identical(self):
        rng = np.random.default_rng(4)
        for b in rng.normal(size=10000) * 1000:
            b = float(b)
            for scale in (.99, 1., 1.01):
                a = b + scale * (1e-8 + 1e-8 * abs(b))
                self.assertEqual(m.close_value(a, b), bool(np.isclose(a, b, rtol=1e-8, atol=1e-8)))
        self.assertFalse(m.close_value(None, 1.))
        self.assertFalse(m.close_value(float('inf'), float('inf')))

    def test_missing_evaluation_days_prevent_target_acceptance(self):
        dates = pd.bdate_range('2020-01-01', periods=5)
        result = {'a': {'requirements': {'prior': True}, 'numericalTargetsMet': True}}
        frame = pd.DataFrame({'date': dates[:2].strftime('%Y-%m-%d')})
        cov = m.coverage_gate(result, frame, dates, {i: {} for i in range(4)}, 0, 5)
        self.assertEqual(len(cov['missingPredictions']), 2)
        self.assertEqual(len(cov['noCommonSupportDates']), 1)
        self.assertFalse(result['a']['numericalTargetsMet'])

    def synthetic(self):
        dates = pd.bdate_range('2020-01-01', periods=143)
        rng = np.random.default_rng(12)
        daily = {}
        for i in range(len(dates)):
            raw = rng.normal(size=(40, 12))
            y = rng.normal(0, .04, 40)
            daily[i] = {'features': m.rank_features(raw), 'symbols': np.array([f'SH.{600000+k:06d}' for k in range(40)]),
                        'y': y, 'entry': np.ones(40, bool), 'delay': np.zeros(40)}
        c = copy.deepcopy(self.c)
        c['training'].update(trainSessions=40, calibrationSessions=20, minimumTrainDays=30,
                             minimumCalibrationDays=15, minimumCrossSection=10)
        c['evaluation']['startDate'] = str(dates[100].date())
        return dates, daily, c

    def test_real_fit_ignores_unmatured_future_labels(self):
        dates, daily, c = self.synthetic()
        view = m.model_daily(daily, 'interaction_model')
        train, cal = m.payoff.split_indices(100, c)
        model, _ = m.fit(view, train, cal, c)
        altered = copy.deepcopy(view)
        for i in range(94, len(dates)):
            altered[i]['y'][:] = 10000
        other, _ = m.fit(altered, train, cal, c)
        a, b = m.predict(model, view[100]['x']), m.predict(other, view[100]['x'])
        np.testing.assert_allclose(a['p'], b['p'], atol=1e-12)
        np.testing.assert_allclose(a['payoff_decomposition'], b['payoff_decomposition'], atol=1e-12)

    def test_real_run_trains_all_arms_fixed_ten_and_no_overwrite(self):
        dates, daily, c = self.synthetic()
        # Only input construction is substituted; actual fitting, calibration,
        # full evaluation and report paths run end-to-end on synthetic data.
        with tempfile.TemporaryDirectory() as tmp:
            c['outputRoot'] = str(Path(tmp) / 'outputs/edge_research/run')
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps(c), encoding='utf-8')
            with patch.object(m, 'validate'), patch.object(m, 'load_data', return_value=(dates, daily, {'syntheticOnly': True})):
                with m.threadpool_limits(limits=1):
                    r = m.run(config, 'smoke')
                self.assertEqual(set(r['arms']), set(m.ARMS))
                self.assertGreaterEqual(r['folds'], 2)
                self.assertFalse(r['mayPromote'])
                self.assertEqual(r['orders'], [])
                p = pd.read_csv(Path(c['outputRoot']) / 'smoke/historical_picks.csv')
                self.assertTrue(p.groupby(['policy', 'date']).size().eq(10).all())
                self.assertTrue(p.pTail.between(0, 1).all())
                self.assertTrue(np.allclose(p.net, p.gross - .003))
                with self.assertRaises(FileExistsError):
                    m.run(config, 'smoke')

    def test_unresolved_selection_remains_in_denominator(self):
        rows = [{'date': '2020-01-02', 'state': 'resolved' if k else 'unfilled_entry',
                 'gross': .02 if k else np.nan, 'net': .017 if k else np.nan, 'pUp': .55, 'pTail': .1} for k in range(10)]
        r = m.summary(pd.DataFrame(rows), self.c)
        self.assertEqual(r['selected'], 10)
        self.assertEqual(r['resolved'], 9)
        self.assertEqual(r['completeDays'], 0)
        self.assertEqual(r['allSelectedUpBounds'], [.9, 1.])
        self.assertTrue(np.isnan(r['netPerPickOnCompleteDays']))


if __name__ == '__main__':
    unittest.main()
