"""Contracts for a fixed candidate, old-data reuse and equivalent training streams."""
from common7 import *
import unittest


class Contracts(unittest.TestCase):
    def test_unused_early_evaluation_does_not_change_training(self):
        values = legacy.batch_tensors(np.arange(16))
        labels, _ = legacy.standardized_targets(np.arange(16))
        for arm in ['baseline', 'combined']:
            results = []
            for intermediate_evaluation in [True, False]:
                network = legacy.make_model(arm, 20260910)
                decay = .01 if arm == 'baseline' else .1
                optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=decay)
                for epoch in range(1, 6):
                    network.train(); optimizer.zero_grad(set_to_none=True)
                    out, aux = legacy.predict(network, values, True)
                    loss = ((out-labels['returns'])**2).mean()+.1*((aux-labels['auxiliary'])**2).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    if intermediate_evaluation and epoch in [2, 5]:
                        network.eval()
                        with torch.inference_mode():
                            legacy.predict(network, values, False)
                        legacy.evaluate_training(network, values, labels)
                results.append((network.state_dict(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
            for key in results[0][0]:
                self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]), (arm, key))
            self.assertTrue(torch.equal(results[0][1], results[1][1]))
            self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def test_masks_maturity_and_three_seed_sampling_budget(self):
        obs = pd.read_csv(OUT / 'observation_table.csv')
        for fold in cfg()['folds']:
            self.assertEqual(sha(CACHE / f'masks_{fold["cutoff"]}.npz'), sha(V6 / 'cache' / f'masks_{fold["cutoff"]}.npz'))
            with np.load(CACHE / f'masks_{fold["cutoff"]}.npz') as masks:
                full = masks['full']; te = masks['testing']
                for history, group in [('drop_early', 0), ('drop_middle', 2), ('drop_recent', 4)]:
                    tr = masks[history]
                    np.testing.assert_array_equal(np.setdiff1d(full, tr), np.array_split(full, 5)[group])
                    self.assertTrue(obs.joint_completed.iloc[tr].le(fold['cutoff']).all())
                    self.assertTrue(obs.date.iloc[te].gt(fold['cutoff']).all())
                    for seed in cfg()['training']['seeds']:
                        rng = np.random.default_rng(seed)
                        for _ in range(20):
                            order = legacy.epoch_order(len(tr), len(full), rng)
                            self.assertEqual(len(order), len(full))
                            self.assertEqual(len(np.unique(order)), len(tr))
                            self.assertTrue(np.isin(tr[order], tr).all())
                            self.assertFalse(np.isin(tr[order], te).any())

    def test_plan_and_reused_candidate_are_frozen(self):
        selected = json.loads((V6 / 'results/selection.json').read_text(encoding='utf-8'))['selected']
        self.assertEqual((selected['arm'], selected['epoch']), ('combined', 20))
        self.assertEqual(role('combined', 20), 'candidate20')
        self.assertEqual(role('baseline', 10), 'baseline10')
        jobs = pd.DataFrame(fit_plan())
        self.assertEqual(len(jobs), 54)
        self.assertFalse(jobs.duplicated().any())
        self.assertTrue(jobs.groupby(['arm', 'history', 'cutoff']).size().eq(3).all())
        self.assertTrue(jobs.groupby(['history', 'cutoff', 'seed']).size().eq(2).all())
        reused = pd.read_csv(OUT / 'reused_full_predictions.csv')
        self.assertEqual(len(reused), 1692)
        self.assertTrue(reused.groupby(['role', 'date']).size().eq(3).all())
        self.assertTrue(reused.history.eq('full').all())
        self.assertFalse(reused[['predicted_return', 'reassigned_prediction']].isna().any().any())


if __name__ == '__main__':
    legacy.initialize()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if result.wasSuccessful():
        save(OUT / 'contract_verification.json', dict(status='PASS', tests=result.testsRun,
             source_sha256=source_hashes(), protocol_sha256=sha(ROOT / 'protocol.json'),
             test_source_sha256=sha(Path(__file__)), checked_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    sys.exit(0 if result.wasSuccessful() else 1)
