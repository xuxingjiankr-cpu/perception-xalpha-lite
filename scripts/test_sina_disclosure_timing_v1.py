import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_sina_disclosure_timing_v1 as m
import test_sina_fundamental_top10_v1 as base_tests


class DisclosureTimingTests(unittest.TestCase):
    def setUp(self):
        self.c = m.read(m.ROOT / 'configs/research/sina_disclosure_timing_v1.json')
        self.fc = {'quality': {'candidates': [{'id': 'q', 'field': 'value', 'transform': 'level', 'orientation': 1., 'scale': 1.}]}}

    def fixture(self):
        helper = base_tests.SinaFundamentalTrainingTests()
        helper.setUp()
        sessions, daily, bc = helper.synthetic()
        for i, row in daily.items():
            age = (np.arange(len(row['symbols'])) + i) % 70
            original = row['features']['interaction_model']
            row['features'].update(pooled_payoff=original,
                                   disclosure_timing=m.timing_matrix(original, age, [5, 20]),
                                   lagged_timing_counter=m.timing_matrix(original, age + 20, [5, 20]))
            row['disclosureAge'] = age
        return sessions, daily, bc

    def test_disclosure_notice_update_and_future_prefix(self):
        dates = pd.bdate_range('2020-01-01', periods=220)
        rows = [{'noticeDate': str(dates[2].date()), 'updateDate': str(dates[4].date()),
                 'reportDate': '2019-12-31', 'value': 1}]
        age, _ = m.disclosure_age(rows, dates, 'SH.600000', self.fc)
        self.assertTrue(np.isnan(age[:5]).all())
        self.assertEqual(age[5], 0)
        self.assertEqual(age[205], 200)  # known historical date does not expire
        future = rows + [{'noticeDate': str(dates[150].date()), 'reportDate': '2020-06-30', 'value': 9}]
        other, _ = m.disclosure_age(future, dates, 'SH.600000', self.fc)
        np.testing.assert_equal(age[:151], other[:151])
        self.assertEqual(other[151], 0)
        absent, _ = m.disclosure_age([{'reportDate': '2019-12-31', 'value': 5}], dates, 'SH.600000', self.fc)
        self.assertTrue(np.isnan(absent).all())

    def test_nonadvancing_report_does_not_rewrite_clock(self):
        dates = pd.bdate_range('2020-01-01', periods=80)
        rows = [{'noticeDate': str(dates[2].date()), 'reportDate': '2019-12-31', 'value': 1},
                {'noticeDate': str(dates[40].date()), 'reportDate': '2019-12-31', 'value': 99}]
        a, audit = m.disclosure_age(rows, dates, 'SH.600000', self.fc)
        self.assertEqual(a[41], 38)
        self.assertEqual(audit['non_advancing_report_date_skipped'], 1)

    def test_fixed_decays_interactions_and_unknown_not_imputed(self):
        raw = np.ones((3, 12)) * .25
        x = m.timing_matrix(raw, np.array([0., 5., 20.]), [5, 20])
        np.testing.assert_equal(x[:, :12], raw)
        np.testing.assert_allclose(x[0, 12:14], [1, 1])
        self.assertEqual(x[1, 12], .5)
        self.assertEqual(x[2, 13], .5)
        np.testing.assert_equal(x[:, 14:16], x[:, 12:14] * .25)
        self.assertEqual(x.shape, (3, 22))
        with self.assertRaisesRegex(ValueError, 'unknown_age'):
            m.timing_matrix(raw, np.array([np.nan, 5, 20]), [5, 20])

    def test_add_timing_uses_exact_twenty_session_lag_same_support(self):
        dates = pd.bdate_range('2020-01-01', periods=100)
        raw = np.ones((10, 12)) * .2
        row = {'symbols': np.array([f'SH.{600000+j}' for j in range(10)]),
               'features': {'interaction_model': raw}, 'y': np.zeros(10), 'entry': np.ones(10, bool), 'delay': np.zeros(10)}
        rows = [{'noticeDate': str(dates[1].date()), 'reportDate': '2019-12-31', 'value': 1},
                {'noticeDate': str(dates[60].date()), 'reportDate': '2020-03-31', 'value': 2}]
        hashes = {f'fake/SH_{600000+j}.jsonl': 'a' for j in range(10)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'read', return_value={'families': self.fc, 'fundamentals': {'root': 'fake'}}), patch.object(m.base.readiness, 'read_rows', return_value=(rows, 'a')):
            r = m.add_timing({65: row}, dates, {'fundamentalConfig': 'ignored'}, self.c, hashes, Path(tmp))[65]
        np.testing.assert_equal(r['symbols'], row['symbols'])
        np.testing.assert_equal(r['y'], row['y'])
        self.assertTrue((r['disclosureAge'] == 4).all())
        self.assertAlmostEqual(r['features']['disclosure_timing'][0, 12], 2 ** (-4 / 5))
        self.assertAlmostEqual(r['features']['lagged_timing_counter'][0, 12], 2 ** (-43 / 5))

    def test_real_numeric_model_roundtrip_and_class_rejection(self):
        _, daily, bc = self.fixture()
        train, cal = m.base.payoff.split_indices(100, bc)
        with m.threadpool_limits(limits=1):
            model, _ = m.base.fit(m.base.model_daily(daily, 'disclosure_timing'), train, cal, bc)
        artifact = m.base.model_parameters(model)
        restored = m.restore_model(json.loads(json.dumps(artifact)))
        a = m.base.predict(model, daily[100]['features']['disclosure_timing'])
        b = m.base.predict(restored, daily[100]['features']['disclosure_timing'])
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
        artifact['parameters']['up']['class'] = 'eval'
        with self.assertRaisesRegex(ValueError, 'untrusted_model'):
            m.restore_model(artifact)

    def test_real_fit_unmatured_future_labels_cannot_change_predictions(self):
        _, daily, bc = self.fixture()
        train, cal = m.base.payoff.split_indices(100, bc)
        other = copy.deepcopy(daily)
        for i in range(94, len(other)):
            other[i]['y'][:] = 500
        with m.threadpool_limits(limits=1):
            a, _ = m.base.fit(m.base.model_daily(daily, 'disclosure_timing'), train, cal, bc)
            b, _ = m.base.fit(m.base.model_daily(other, 'disclosure_timing'), train, cal, bc)
        self.assertEqual(m.base.model_parameters(a), m.base.model_parameters(b))

    def make_reference(self, dates, daily, bc, out):
        picks = []
        end = len(dates) - bc['training']['purgeSessions']
        start = int(dates.searchsorted(bc['evaluation']['startDate']))
        for first in range(start, end, bc['training']['refitSessions']):
            train, cal = m.base.payoff.split_indices(first, bc)
            models = {a: m.base.fit(m.base.model_daily(daily, a), train, cal, bc)[0] for a in ['fundamental_model', 'interaction_model']}
            m.save(out / 'models' / f'fold_{first}.json', {'researchOnly': True, 'firstSignalIndex': first, 'arms': {a: m.base.model_parameters(v) for a, v in models.items()}})
            for i in range(first, min(first + bc['training']['refitSessions'], end)):
                row = daily[i]
                for a in ['equal_families', 'interaction_model']:
                    key = 'fundamental_model' if a == 'equal_families' else a
                    p = m.base.predict(models[key], row['features'][key])
                    score = row['features'][key].mean(axis=1) if a == 'equal_families' else p['payoff_decomposition'] - .003
                    rr = m.base.payoff.record_picks(row, i, dates[i], len(dates) - 1, 'reused_history', a, score, p, bc)
                    for r, k in zip(rr, m.base.payoff.top10(score)):
                        r['pTail'] = p['pTail'][k]
                    picks.extend(rr)
        pd.DataFrame(picks).to_csv(out / 'historical_picks.csv', index=False)

    def test_actual_evaluate_fits_reproduces_calibrates_and_keeps_ten(self):
        dates, daily, bc = self.fixture()
        daily[105]['y'][:] = np.nan
        daily[105]['entry'][:] = False  # must not delete that date or replace its picks
        with tempfile.TemporaryDirectory() as tmp, m.threadpool_limits(limits=1):
            ref, out = Path(tmp) / 'reference', Path(tmp) / 'new'
            ref.mkdir(); out.mkdir()
            self.make_reference(dates, daily, bc, ref)
            r = m.evaluate(dates, daily, bc, self.c, ref, out)
            self.assertEqual(set(r['arms']), set(m.ARMS))
            self.assertEqual(r['orders'], [])
            self.assertFalse(r['mayPromote'])
            self.assertEqual(r['folds'], 2)
            self.assertTrue(r['baselineReproduction']['identicalRankAndSecurity'])
            self.assertEqual(r['jointCompleteComparison']['jointCompleteDays'], 35)
            frame = pd.read_csv(out / 'historical_picks.csv')
            self.assertTrue(frame.groupby(['date', 'policy']).size().eq(10).all())
            self.assertTrue(frame.pTail.between(0, 1).all())
            self.assertTrue(frame.loc[frame.date.eq(str(dates[105].date()))].state.eq('unfilled_entry').all())
            self.assertTrue(all(not v['numericalTargetsMet'] for v in r['comparisons'].values()))
            pm = pd.read_csv(out / 'same_support_probability_metrics.csv')
            self.assertTrue(pm.groupby(['date', 'task'])['count'].nunique().eq(1).all())
            changed = frame.copy()
            changed.loc[0, 'securityId'] = 'SH.999999'
            with self.assertRaisesRegex(ValueError, 'baseline_selection'):
                m.verify_baseline(changed, pd.read_csv(ref / 'historical_picks.csv'))

    def test_shared_probability_support_no_return_based_selection(self):
        _, daily, bc = self.fixture()
        row = daily[100]
        row['y'][0] = np.nan
        predictions = {a: {'p': np.full(40, .51), 'pTail': np.full(40, .15)} for a in m.ARMS}
        result = m.probability_rows(row, pd.Timestamp('2020-01-01'), predictions, bc)
        self.assertEqual({r['count'] for r in result}, {39})
        predictions['disclosure_timing']['p'][2] = np.nan
        with self.assertRaisesRegex(ValueError, 'no_support_filtering'):
            m.probability_rows(row, pd.Timestamp('2020-01-01'), predictions, bc)

    def test_preregistration_hash_contract_and_output_no_overwrite(self):
        m.validate(self.c)
        changed = copy.deepcopy(self.c)
        changed['features']['halfLifeSessions'] = [3, 7]
        with self.assertRaisesRegex(ValueError, 'preregistration'):
            m.validate(changed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The test must work without private historical outputs. Only the
            # reference-contract fixture is synthetic; digest checks are real.
            bc = m.read(m.ROOT / self.c['baseConfig'])
            config_copy = root / self.c['baseConfig']
            config_copy.parent.mkdir(parents=True)
            config_copy.write_bytes((m.ROOT / self.c['baseConfig']).read_bytes())
            ref = root / 'outputs/edge_research/sina_fundamental_top10_v1' / self.c['referenceRun']
            ref.mkdir(parents=True)
            m.save(ref / 'manifest.json', {'config': bc, 'dependencies': {}, 'numpy': np.__version__,
                   'pandas': pd.__version__, 'sklearn': m.sklearn.__version__})
            m.save(ref / 'status.json', {'state': 'completed_diagnostic_only'})
            m.save(ref / 'input_hashes.json', [])
            m.base.source.atomic_text(ref / 'historical_picks.csv', 'date,policy\n')
            m.save(ref / 'models/fold_1.json', {'syntheticHashFixtureOnly': True})
            fixture = copy.deepcopy(self.c)
            fixture['referenceManifestSha256'] = m.base.pit.digest(ref / 'manifest.json')
            with patch.object(m, 'ROOT', root):
                _, _, hashes = m.reference_contract(fixture)
                self.assertEqual(len(hashes), 6)
                changed = copy.deepcopy(fixture)
                changed['baseConfigSha256'] = 'bad'
                with self.assertRaisesRegex(ValueError, 'reference_changed'):
                    m.reference_contract(changed)
            path = root / 'config.json'
            path.write_text(json.dumps(self.c), encoding='utf-8')
            (root / self.c['outputRoot'] / 'existing').mkdir(parents=True)
            with patch.object(m, 'ROOT', root), patch.object(m, 'reference_contract', return_value=(bc, ref, hashes)), patch.object(m.base, 'load_data') as loader:
                with self.assertRaises(FileExistsError):
                    m.run(path, 'existing')
                loader.assert_not_called()

    def test_missing_support_and_duplicate_hashes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_input_hash'):
            m.normalized_hashes([{'path': 'a/b', 'sha256': '1'}, {'path': 'a\\b', 'sha256': '1'}])
        dates = pd.bdate_range('2020-01-01', periods=4)
        r = {'test': {'requirements': {'prior': True}, 'numericalTargetsMet': True}}
        m.base.coverage_gate(r, pd.DataFrame({'date': [str(dates[0].date())]}), dates, {0: {}, 1: {}}, 0, 4)
        self.assertFalse(r['test']['numericalTargetsMet'])


if __name__ == '__main__':
    unittest.main()
