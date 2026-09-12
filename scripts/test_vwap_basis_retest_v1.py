import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_vwap_basis_retest_v1 as m
import research_perception_xalpha_rolling_health_v4 as rolling


class VwapBasisRetestTests(unittest.TestCase):
    @staticmethod
    def fixture(n=300, width=30):
        rng = np.random.default_rng(17)
        dates = pd.bdate_range('2020-01-02', periods=n)
        codes = [f'SH.{600000+i}' for i in range(width)]
        raw = 15 * np.exp(np.cumsum(rng.normal(0, .015, (n, width)), axis=0))
        multipliers = np.linspace(1.3, 7.678, width)[None, :] * np.linspace(1, 1.2, n)[:, None]
        frame = lambda x: pd.DataFrame(x, index=dates, columns=codes)
        close = frame(raw * multipliers)
        panel = {'close': close, 'open': close * 1.001, 'high': close * 1.02, 'low': close * .98,
                 'volume': frame(rng.uniform(1000, 9000, (n, width))), 'eligible': close.notna(),
                 'is_st': close * 0, 'trade_status': close * 0 + 1}
        panel['vwap'] = (panel['close'] + panel['open'] + panel['high'] + panel['low']) / 4
        panel['amount'] = panel['volume'] * frame(raw * 1.00025)
        panel['returns'] = close.pct_change(fill_method=None)
        panel['membership'] = close.notna()
        return panel

    def test_adjustment_basis_regression_ratio_7678_must_fail(self):
        p = self.fixture(5, 3)
        good = m.precision.build_factor_inputs(p)
        self.assertTrue(good['vwap'].equals(p['vwap']))
        self.assertEqual(m.price_basis_audit(p, good, require_consistent=True)['outsideAdjustedRange'], 0)
        bad = m.precision.build_factor_inputs(p, vwap_basis=m.OLD)
        with self.assertRaisesRegex(ValueError, 'not_on_ohlc_price_basis'):
            m.price_basis_audit(p, bad, require_consistent=True)
        self.assertGreater((good['vwap'] / bad['vwap']).max().max(), 7.678)
        self.assertNotIn('factorInputBasisAudit', p['vwap'].attrs)  # no input mutation

    def test_absent_fallback_is_explicit_and_not_certified(self):
        p = self.fixture(5, 3)
        p['vwap'].iloc[0, 0] = np.nan
        p['vwap'].iloc[1, 0] = np.nan
        p['volume'].iloc[1, 0] = 0
        v = m.precision.build_factor_inputs(p)['vwap']
        self.assertEqual(v.iloc[0, 0], p['amount'].iloc[0, 0] / p['volume'].iloc[0, 0])
        self.assertEqual(v.iloc[1, 0], p['close'].iloc[1, 0])
        self.assertEqual(v.attrs['factorInputBasisAudit']['unadjustedAmountVolumeFallbackCells'], 1)
        self.assertEqual(v.attrs['factorInputBasisAudit']['closeFallbackCells'], 1)
        p.pop('vwap')
        self.assertEqual(m.precision.build_factor_inputs(p)['vwap'].attrs['factorInputBasisAudit']['archivePreservedCells'], 0)

    def test_feature_prefix_and_archive_not_overwritten_by_amount(self):
        p = self.fixture(60, 3)
        before = m.precision.build_factor_inputs(p)
        after = {k: v.copy() for k, v in p.items()}
        for k in ['close', 'open', 'high', 'low', 'amount', 'volume', 'vwap']:
            after[k].iloc[40:] *= 100
        new = m.precision.build_factor_inputs(after)
        pd.testing.assert_frame_equal(before['vwap'].iloc[:40], new['vwap'].iloc[:40])
        pd.testing.assert_frame_equal(before['returns'].iloc[:40], new['returns'].iloc[:40])

    def test_real_twelve_only_alpha094_changes_and_same_support(self):
        p = self.fixture()
        p['eligible'].iloc[:65] = False
        frozen = m.read(m.ROOT / 'configs/research/perception_xalpha_horizon_precision_v3.json')
        old, _, _ = rolling.compute_rank_book(p, frozen, vwap_basis=m.OLD)
        new, _, _ = rolling.compute_rank_book(p, frozen, vwap_basis=m.NEW)
        common, audit = m.common_rank_support(p, old, new)
        self.assertEqual(sum(r['unchanged'] for r in audit['factorChanges']), 11)
        pd.testing.assert_frame_equal(common, p['eligible'])
        self.assertEqual(audit['droppedCells'], 0)
        a, b = m.paired_ranks(old, common), m.paired_ranks(new, common)
        for k in a:
            pd.testing.assert_frame_equal(a[k], old[k])
            pd.testing.assert_frame_equal(b[k], new[k])
        # A missing factor must not remove a name with other available factors.
        incomplete_old, incomplete_new = copy.deepcopy(old), copy.deepcopy(new)
        for book in (incomplete_old, incomplete_new):
            book['qlib158/vstd60'].iloc[-1, 0] = np.nan
        unchanged, _ = m.common_rank_support(p, incomplete_old, incomplete_new)
        pd.testing.assert_frame_equal(unchanged, p['eligible'])
        unsupported = copy.deepcopy(old)
        for frame in unsupported.values():
            frame.iloc[-1, 0] = np.nan
        # Align the other eleven first, so this pins coverage rather than the
        # independent one-factor-only assertion.
        with self.assertRaisesRegex(ValueError, 'not_fully_supported'):
            m.common_rank_support(p, unsupported, unsupported)
        changed = copy.deepcopy(new)
        changed['qlib158/vstd60'].iloc[-1, 0] += .1
        with self.assertRaisesRegex(ValueError, 'outside_single_vwap_factor'):
            m.common_rank_support(p, old, changed)

    def test_rv20_gap_is_bounded_and_recorded_not_a_silent_drop(self):
        p = self.fixture()
        p['eligible'].iloc[:65] = False
        frozen = m.read(m.ROOT / 'configs/research/perception_xalpha_horizon_precision_v3.json')
        old, _, _ = rolling.compute_rank_book(p, frozen, vwap_basis=m.OLD)
        new, _, _ = rolling.compute_rank_book(p, frozen, vwap_basis=m.NEW)

        # A clean panel must still report a zero tail, and the default bound of
        # zero must accept it, so the relaxation is inert unless a gap exists.
        common, audit = m.common_rank_support(p, old, new)
        self.assertEqual(audit['rv20MissingCells'], 0)
        self.assertEqual(audit['rv20MissingShare'], 0.0)
        self.assertEqual(audit['rv20MissingSymbols'], 0)
        self.assertEqual(audit['droppedCells'], 0)
        pd.testing.assert_frame_equal(common, p['eligible'])

        # A suspension: blank one name's returns so its trailing window cannot
        # reach min_periods. It stays eligible and must stay in the support.
        halted = copy.deepcopy(p)
        halted['returns'].iloc[70:95, 0] = np.nan
        common, audit = m.common_rank_support(halted, old, new, 0.005)
        self.assertGreater(audit['rv20MissingCells'], 0)
        self.assertEqual(audit['rv20MissingSymbols'], 1)
        self.assertEqual(audit['droppedCells'], 0)
        # Recorded, never removed: eligibility is untouched and nothing reranks.
        pd.testing.assert_frame_equal(common, halted['eligible'])
        for k in old:
            pd.testing.assert_frame_equal(m.paired_ranks(old, common)[k], old[k])

        # The bound is real: the same gap under the default zero share refuses.
        with self.assertRaisesRegex(ValueError, 'rv20_missing_share_above_preregistered_threshold'):
            m.common_rank_support(halted, old, new)

        # And a gap wider than the preregistered share refuses too.
        wide = copy.deepcopy(p)
        wide['returns'].iloc[70:, :] = np.nan
        with self.assertRaisesRegex(ValueError, 'rv20_missing_share_above_preregistered_threshold'):
            m.common_rank_support(wide, old, new, 0.005)

    def test_rv20_is_identical_across_price_bases(self):
        # This is what licenses tolerating an rv20 gap at all: rv20 comes from
        # returns, so it cannot differ between the two VWAP bases and cancels
        # exactly in the paired comparison. If a future edit ever derives rv20
        # from a price-basis-dependent field, this fails.
        p = self.fixture()
        rv_old = m.np.isfinite(
            p['returns'].rolling(20, min_periods=10).std())
        inputs_old = m.precision.build_factor_inputs(p, vwap_basis=m.OLD)
        inputs_new = m.precision.build_factor_inputs(p, vwap_basis=m.NEW)
        self.assertFalse(inputs_old['vwap'].equals(inputs_new['vwap']))
        for inputs in (inputs_old, inputs_new):
            rv = m.np.isfinite(inputs['returns'].rolling(20, min_periods=10).std())
            pd.testing.assert_frame_equal(rv, rv_old)

    def test_config_contract_pins_the_rv20_bound(self):
        for name in ('vwap_basis_retest_v1.json', 'vwap_basis_retest_v2.json'):
            path = m.ROOT / 'configs/research' / name
            if not path.exists():
                continue
            c = m.read(path)
            self.assertEqual(c['supportVersion'],
                             'preserve_original_eligibility_rv20_bounded_v3')
            self.assertIsInstance(c['rv20MissingMaxShare'], float)
            self.assertGreaterEqual(c['rv20MissingMaxShare'], 0.0)
            self.assertLessEqual(c['rv20MissingMaxShare'], 0.01)
            # A bound this loose would stop being a bound.
            self.assertLess(c['rv20MissingMaxShare'], 0.01)
            self.assertFalse(c['mayPromote'])
            self.assertFalse(c['mayTrade'])
            self.assertEqual(c['orders'], [])

    def test_cache_invalidates_basis_and_input_builder(self):
        c = m.panel_cache
        frozen = {'frozenFactors': [{'factorKey': 'z/test'}]}
        old = c.rank_book_key('p', frozen, vwap_basis=m.OLD)
        new = c.rank_book_key('p', frozen, vwap_basis=m.NEW)
        self.assertNotEqual(old, new)
        self.assertIn('research_perception_xalpha_horizon_precision_v3.py', c.RANK_BOOK_SOURCES)
        with patch.object(c, 'RANK_BOOK_SOURCES', ('research_perception_xalpha_rolling_health_v4.py',)):
            self.assertNotEqual(new, c.rank_book_key('p', frozen))
        panel = self.fixture(20, 3)
        with tempfile.TemporaryDirectory() as tmp, patch.object(rolling, 'compute_rank_book', return_value=({'z/test': panel['close']}, panel['close'], {'factorCount': 1})) as compute:
            for basis in (m.OLD, m.NEW, m.OLD, m.NEW):
                c.build_rank_book_cached(panel, frozen, 'p', cache_root=Path(tmp), verbose=False, vwap_basis=basis)
            self.assertEqual(compute.call_count, 2)

    def test_vstd60_is_volume_not_price_volatility(self):
        mod = importlib.import_module('src.factors.zoo.qlib158.vstd60')
        p = self.fixture()
        old = mod.compute(p)
        p['close'] *= 20
        p['returns'] *= 4
        pd.testing.assert_frame_equal(old, mod.compute(p))

    def test_both_actual_studies_paired_paths_no_old_output_or_overwrite(self):
        panel = self.fixture(180)
        frozen = m.read(m.ROOT / 'configs/research/perception_xalpha_horizon_precision_v3.json')
        ranks = {x['factorKey']: panel['close'].rank(axis=1, pct=True) for x in frozen['frozenFactors']}
        common = panel['eligible'].copy()
        common.iloc[:, -1] = False
        ranks = m.paired_ranks(ranks, common)
        dates = panel['close'].index
        source = {'splitAudit': {'train': [str(dates[0].date()), str(dates[59].date())],
                                'validation': [str(dates[60].date()), str(dates[119].date())],
                                'shadowQuarantine': [str(dates[120].date()), str(dates[-1].date())]}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Patch only external source lookup/output root, not study arithmetic.
            with patch.object(m.guarded, 'load_frozen_config', return_value=(frozen, source, 'fixture')), patch.object(m.perception, 'load_base_configs', return_value=({}, {})):
                for mod, cfg in [(m.frontier, 'configs/research/horizon_cost_frontier_v1.json'), (m.tail, 'configs/research/tail_exclusion_screen_v1_volatility_ablation.json')]:
                    config = m.read(m.ROOT / cfg)
                    # Preserve real validators/configs; substitute only base file lookup.
                    original_load = mod.load_json
                    with patch.object(mod, 'ROOT', root), patch.object(mod.precision, 'ROOT', root), patch.object(mod, 'load_json', side_effect=lambda path: original_load(path) if Path(path).exists() else frozen):
                        prepare = {'output': 'outputs/edge_research/vwap_basis_retest_v1/synthetic/' + mod.__name__,
                                   'panel': panel, 'panelAudit': {}, 'ranks': ranks, 'factorAudit': {}, 'commonSupport': common,
                                   'audit': {'researchOnly': True, 'supportSha256': m.frame_digest(common)}}
                        # Config absolute path was captured before the temporary root substitution.
                        result = mod.run(Path(__file__).resolve().parents[1] / cfg, 'synthetic', paired_inputs=prepare)
                        self.assertFalse(result['verdict']['eligibleForTrading'])
                        self.assertEqual(result['orders'], [])
                        self.assertEqual(result['basisRetest']['supportSha256'], m.frame_digest(common))
                        with self.assertRaises(FileExistsError):
                            mod.run(Path(__file__).resolve().parents[1] / cfg, 'synthetic', paired_inputs=prepare)

    def test_rejection_gate_requires_all_cells_and_both_axes(self):
        c = m.read(m.ROOT / 'configs/research/vwap_basis_retest_v1.json')
        f = {'periods': {p: {'headline': [{'book': b, 'holdingTradingDays': 1, 'meanExcessPerPick': .001, 'excessDayClusteredT': 3., 'meanGrossPerPick': .002} for b in c['compositeBooks']]} for p in c['tailRequiredWindows']}}
        cells = []
        for p in c['tailRequiredWindows']:
            for h in [1, 5]:
                for b in c['compositeBooks'] + ['realized_volatility_20', 'single/qlib158/vstd60']:
                    cells.append({'book': b, 'period': p, 'holdingTradingDays': h, 'bucket': 10,
                                  'excessSevereLossRate': .2 if b in c['compositeBooks'] else .1,
                                  'meanExcessReturn': -.02 if b in c['compositeBooks'] else -.01})
        t = {'cells': cells, 'bucketCount': 10}
        r = m.conclusions(f, f, t, t, c)
        self.assertTrue(r['study12']['standingConclusionRejected'])
        self.assertTrue(r['study14']['standingConclusionRejected'])
        for row in f['periods']['shadow']['headline']:
            row['excessDayClusteredT'] = 1.7
        for row in t['cells']:
            if row['period'] == 'shadow' and row['holdingTradingDays'] == 5 and row['book'] in c['compositeBooks']:
                row['meanExcessReturn'] = .02  # better tail alone must fail
        r = m.conclusions(f, f, t, t, c)
        self.assertFalse(r['study12']['standingConclusionRejected'])
        self.assertFalse(r['study14']['standingConclusionRejected'])
        self.assertFalse(r['mayPromote'])


if __name__ == '__main__':
    unittest.main()
