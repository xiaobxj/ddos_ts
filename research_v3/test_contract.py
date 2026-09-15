import unittest
from common import *


class ResearchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(V1 / 'data/1_000300.csv')
        cls.obs = observation_table(cls.frame)
        cls.panels = [pd.read_csv(V1 / 'data' / f'{s}.csv') for s in cfg()['cross_index_features']['indices']]

    def test_labels_identical_to_previous_round_and_holiday_maturity(self):
        previous = pd.read_csv(V2 / 'results/observation_table.csv')
        pd.testing.assert_frame_equal(self.obs, previous, check_dtype=False)
        row = self.obs[self.obs.date == '2020-12-25'].iloc[0]
        self.assertEqual(row.exit_date, '2021-01-11')
        tr, _ = masks(self.obs, '2020-12-31', '2021-03-31')
        self.assertFalse(tr[row.name])

    def test_previous_feature_and_ridge_parity(self):
        ref = np.load(ROOT / 'sources/v2_parity_reference.npz')
        actual = []
        for t in ref['anchors']:
            channels, base = economic_window(self.frame.iloc[t-124:t+1])
            actual.append(multi_features(channels, base))
        np.testing.assert_allclose(actual, ref['features'], atol=1e-12, rtol=0)
        result = ridge_path(ref['x'], ref['y'], ref['xt'], [0.1, 10.0])
        np.testing.assert_allclose(result['0.1'], ref['ridge01'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(result['10.0'], ref['ridge10'], atol=1e-12, rtol=0)

    def test_target_and_panel_future_perturbation_invariance(self):
        selected = self.obs[(self.obs.anchor >= 3000) & (self.obs.anchor <= 3003)].copy()
        target, extra, valid = build_features(self.frame, selected, self.panels)
        changed = self.frame.copy()
        cutoff = selected.date.max()
        cols = ['open', 'high', 'low', 'close', 'volume']
        changed.loc[changed.date > cutoff, cols] *= 1000.
        panels = [p.copy() for p in self.panels]
        for p in panels:
            p.loc[p.date > cutoff, cols] *= 1000.
        target2, extra2, valid2 = build_features(changed, selected, panels)
        np.testing.assert_array_equal(valid, valid2)
        np.testing.assert_array_equal(target, target2)
        np.testing.assert_array_equal(extra, extra2)

    def test_panel_missing_and_invalid_bars_are_not_filled(self):
        selected = self.obs[self.obs.anchor == 3000].copy()
        date = self.frame.date.iloc[2990]
        panels = [p.copy() for p in self.panels]
        panels[0] = panels[0][panels[0].date != date]
        _, extra, valid = build_features(self.frame, selected, panels)
        self.assertFalse(valid[0]); self.assertTrue(np.isnan(extra).all())
        panels = [p.copy() for p in self.panels]
        panels[0].loc[panels[0].date == date, 'valid_ohlc'] = False
        self.assertFalse(build_features(self.frame, selected, panels)[2][0])

    def test_cross_features_price_scale_invariance(self):
        target = self.frame.close.to_numpy()[2876:3001]
        dates = self.frame.date.iloc[2876:3001]
        panel = np.array([p.set_index('date').loc[dates].close.to_numpy() for p in self.panels])
        a = cross_window(target, panel)
        b = cross_window(target * 3, panel * np.arange(2, 9)[:, None])
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=0)

    def test_kronos_mapping_uses_only_predicted_opens(self):
        from run_kronos import execution_forecast, prepare_input
        path = pd.DataFrame({'open': [100., 90., 110.]})
        self.assertAlmostEqual(execution_forecast(path), 0.1)
        t = 3000
        x, times = prepare_input(self.frame.iloc[:t+1], t, False)
        self.assertEqual(len(x), 125)
        self.assertTrue((x.amount == 0).all())
        np.testing.assert_array_equal(x.volume, self.frame.volume.iloc[t-124:t+1])
        self.assertEqual(times.iloc[-1], pd.Timestamp(self.frame.date.iloc[t]))
        xo, _ = prepare_input(self.frame.iloc[:t+1], t, True)
        self.assertTrue((xo[['volume', 'amount']] == 0).all().all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
