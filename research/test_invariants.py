"""Tests for leakage, alignment, DTW, and patch preservation (not predictive performance)."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from segmentation import dtw_cost, ending_costs, reverse_lengths, fixed_lengths, encode_patches, zscore


class ResearchInvariants(unittest.TestCase):
    def test_dtw_hand_computed(self):
        self.assertEqual(dtw_cost(np.array([0., 1.]), np.array([0., 2.])), (1., 2))
        self.assertEqual(dtw_cost(np.array([0., 0., 1.]), np.array([0., 1.]))[0], 0.)

    def test_all_history_covered_and_residual_retained(self):
        rng = np.random.default_rng(12)
        close = np.exp(rng.normal(0, .01, 200).cumsum())
        p = np.array([zscore(rng.normal(size=15)) for _ in range(3)])
        for cost in ending_costs(close, p):
            for end in [125, 126, 169, 200]:
                ls = reverse_lengths(cost, end)
                self.assertEqual(ls.sum(), 125)
                self.assertLessEqual(len(ls), 25)
                self.assertTrue((ls > 0).all() and (ls <= 25).all())
                self.assertTrue((ls[1:] >= 5).all())

    def test_future_mutation_cannot_change_features(self):
        rng = np.random.default_rng(45)
        a = np.exp(rng.normal(0, .01, 210).cumsum())
        p = np.array([zscore(rng.normal(size=15)) for _ in range(2)])
        b = a.copy()
        b[170:] *= np.exp(rng.normal(0, 4, len(b)-170))
        ca, cb = ending_costs(a, p), ending_costs(b, p)
        for xa, xb in zip(ca, cb):
            np.testing.assert_array_equal(xa[:171], xb[:171])
            np.testing.assert_array_equal(reverse_lengths(xa, 170), reverse_lengths(xb, 170))

    def test_mask_duration_endpoints_and_column_order(self):
        x = np.arange(125 * 6, dtype=float).reshape(125, 6)
        lengths = fixed_lengths(125, 20)
        encoded = encode_patches(x, lengths).reshape(25, 33)
        np.testing.assert_array_equal(encoded[-1, 24:30], x[-1])
        self.assertEqual(encoded[:, -1].sum(), len(lengths))
        self.assertAlmostEqual(encoded[:, -3].sum(), 1.)
        self.assertEqual(encoded[-1, -2], 1.)
        self.assertTrue((encoded[:-len(lengths)] == 0).all())
        self.assertEqual(encoded[-1, 27], x[-1, 3])

    def test_saved_folds_are_purged(self):
        file = Path(__file__).resolve().parent / "results" / "folds.json"
        if not file.exists():
            self.skipTest("walk-forward results not yet generated")
        for fold in json.loads(file.read_text()):
            self.assertLessEqual(fold["last_training_label"], fold["train_cutoff"])
            self.assertLess(fold["train_cutoff"], fold["test_start"])
            self.assertLessEqual(fold["prototype_cutoff"], fold["train_cutoff"])

    def test_execution_uses_next_open_and_charges_entry_exit(self):
        from evaluate import backtest
        frame = pd.DataFrame(dict(date=["2021-01-08", "2021-01-11", "2021-01-12"], open=[100., 110., 121.]))
        signal = pd.DataFrame(dict(anchor=[0], position=[1]))
        metrics, _ = backtest(frame, signal, 10)
        self.assertAlmostEqual(metrics["total_return"], 1.1*.999*.999-1)
        self.assertEqual(metrics["one_way_turnover"], 2)
        self.assertEqual(metrics["start"], "2021-01-11")

    def test_cash_stays_cash_and_never_pays_fees(self):
        from evaluate import backtest
        frame = pd.DataFrame(dict(date=["2021-01-08", "2021-01-11", "2021-01-12"], open=[100., 110., 1.]))
        signal = pd.DataFrame(dict(anchor=[0], position=[0]))
        metrics, _ = backtest(frame, signal, 20)
        self.assertEqual(metrics["total_return"], 0.)
        self.assertEqual(metrics["one_way_turnover"], 0)

    def test_persistent_position_does_not_reenter_every_week(self):
        from evaluate import backtest
        frame = pd.DataFrame(dict(date=["2021-01-08", "2021-01-11", "2021-01-12", "2021-01-13"], open=[100.,100.,110.,121.]))
        signal = pd.DataFrame(dict(anchor=[0,1], position=[1,1]))
        metrics, _ = backtest(frame, signal, 10)
        self.assertAlmostEqual(metrics["total_return"], 1.21*.999*.999-1)
        self.assertEqual(metrics["one_way_turnover"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
