"""Behavioral contracts for causal sample masks and the controlled model changes."""
from models6 import *
import unittest
import importlib.util


sys.path.insert(0, str(V5))
spec = importlib.util.spec_from_file_location('round5_reference_models', V5 / 'models5.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class Contracts(unittest.TestCase):
    def test_baseline_weights_rng_and_training_step_match_round5(self):
        seed = cfg()['training']['seeds'][0]
        previous = old.make_model('raw', 'joint_ohlcv', seed)
        cpu = torch.get_rng_state().clone()
        gpu = torch.cuda.get_rng_state().clone()
        current = make_model('baseline', seed)
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu, torch.cuda.get_rng_state()))
        self.assertEqual(previous.state_dict().keys(), current.state_dict().keys())
        for key, value in previous.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[key]), key)
        ids = np.arange(16)
        values = batch_tensors(ids)
        labels, scales = standardized_targets(ids)
        old_labels, old_scales = old.standardized_targets(ids)
        self.assertEqual(scales, old_scales)
        for key in labels:
            self.assertTrue(torch.equal(labels[key], old_labels[key]))
        final_states = []
        for model in [previous, current]:
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(gpu)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            output, auxiliary = model.outputs(**values, auxiliary=True)
            loss = ((output-labels['returns'])**2).mean() + .1*((auxiliary-labels['auxiliary'])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            final_states.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
        for key, value in previous.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[key]), key)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*final_states)))

    def test_width_and_regularization_are_configured_as_frozen(self):
        counts = {}
        values = batch_tensors(np.arange(8))
        labels, _ = standardized_targets(np.arange(8))
        for arm in cfg()['arms']:
            model = make_model(arm['name'], 20260910)
            counts[arm['name']] = sum(p.numel() for p in model.parameters())
            self.assertEqual(model.geometry.out_features, arm['d_model'])
            self.assertTrue(all(m.p == arm['dropout'] for m in model.modules() if isinstance(m, nn.Dropout)))
            output, auxiliary = predict(model, values, True)
            self.assertEqual(tuple(output.shape), (8,))
            self.assertEqual(tuple(auxiliary.shape), (8, 25))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            loss = ((output-labels['returns'])**2).mean() + .1*((auxiliary-labels['auxiliary'])**2).mean()
            loss.backward()
            norm = sum(float(p.grad.square().sum()) for p in model.backbone.parameters() if p.grad is not None)
            self.assertGreater(norm, 0.)
        self.assertEqual(counts['baseline'], 138471)
        self.assertLess(counts['small'], counts['baseline'] / 2)
        self.assertEqual(counts['small'], counts['combined'])
        self.assertEqual(counts['baseline'], counts['dropout'])
        self.assertEqual(counts['baseline'], counts['decay'])

    def test_temporal_masks_and_shared_validation(self):
        obs = pd.read_csv(OUT / 'observation_table.csv')
        total = 0
        for fold, expected in zip(cfg()['folds'], [(1678, 49), (1921, 46), (2165, 46)]):
            histories, te = fold_indices(obs, fold)
            self.assertEqual((len(histories['full']), len(te)), expected)
            total += len(te)
            full = histories['full']
            for name, group in [('drop_early', 0), ('drop_middle', 2), ('drop_recent', 4)]:
                removed = np.setdiff1d(full, histories[name])
                np.testing.assert_array_equal(removed, np.array_split(full, 5)[group])
                self.assertTrue(obs.joint_completed.iloc[histories[name]].le(fold['cutoff']).all())
                self.assertFalse(np.isin(histories[name], te).any())
            with np.load(CACHE / f'masks_{fold["cutoff"]}.npz') as saved:
                for name, tr in histories.items():
                    np.testing.assert_array_equal(saved[name], tr)
                np.testing.assert_array_equal(saved['testing'], te)
        self.assertEqual(total, 141)

    def test_sampling_keeps_retained_only_and_equal_optimizer_budget(self):
        for count in [1678, 1921, 2165]:
            first = np.random.default_rng(20260910)
            second = np.random.default_rng(20260910)
            for _ in range(20):
                np.testing.assert_array_equal(epoch_order(count, count, first), second.permutation(count))
            retained = count - len(np.array_split(np.arange(count), 5)[0])
            first = np.random.default_rng(20260910)
            second = np.random.default_rng(20260910)
            for _ in range(10):
                draws = epoch_order(retained, count, first)
                np.testing.assert_array_equal(draws, epoch_order(retained, count, second))
                self.assertEqual(len(draws), count)
                self.assertEqual(len(np.unique(draws)), retained)
                self.assertTrue(((draws >= 0) & (draws < retained)).all())
                self.assertLessEqual(np.bincount(draws).max(), 2)
                self.assertEqual(len(draws[-(count % 128):]), count % 128)


if __name__ == '__main__':
    initialize()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        save(OUT / 'contract_verification.json', dict(status='PASS', tests=result.testsRun,
             protocol_sha256=sha(ROOT / 'protocol.json'), source_sha256=source_hashes(),
             test_source_sha256=sha(Path(__file__)), checked_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    sys.exit(0 if result.wasSuccessful() else 1)
