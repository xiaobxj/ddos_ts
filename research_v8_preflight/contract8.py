"""Pre-fit checks of the new sampling, weighting and optimizer contracts."""
from common8 import *
from linear8 import fit_ridge
import unittest


class Contracts(unittest.TestCase):
    def test_policies_maturity_nonoverlap_and_budget(self):
        obs = pd.read_csv(OUT / 'observation_table.csv')
        altered = obs.copy(); altered['exec_return'] = np.arange(len(obs))[::-1]
        for fold in cfg()['folds']:
            full, _, te, n = load_policy(fold['cutoff'], 'full')
            expected = np.flatnonzero(obs.joint_completed.le(fold['cutoff']).to_numpy())
            np.testing.assert_array_equal(full, expected)
            for policy in cfg()['policies']:
                tr, weights, testing, full_n = load_policy(fold['cutoff'], policy)
                self.assertEqual(full_n, n); np.testing.assert_array_equal(te, testing)
                self.assertTrue(np.isin(tr, full).all())
                self.assertTrue(obs.joint_completed.iloc[tr].le(fold['cutoff']).all())
                self.assertEqual(len(tr), len(np.unique(tr)))
                self.assertTrue((np.diff(tr)>0).all())
                self.assertAlmostEqual(weights.mean(), 1., places=12)
                tr2, w2 = define_policy(altered, full, policy)
                np.testing.assert_array_equal(tr2, tr); np.testing.assert_array_equal(w2, weights)
                if policy.startswith('disjoint'):
                    ends = np.maximum(obs.exit.iloc[tr].to_numpy(), obs.anchor.iloc[tr].to_numpy()+5)
                    self.assertTrue((obs.entry.iloc[tr[1:]].to_numpy()>ends[:-1]).all())
                    occupied = set()
                    for i, end in zip(tr, ends):
                        support = set(range(int(obs.entry.iloc[i]), int(end)+1))
                        self.assertFalse(occupied & support); occupied |= support
                elif policy=='recent3y':
                    threshold = f'{int(fold["cutoff"][:4])-2}-01-01'
                    np.testing.assert_array_equal(tr, full[obs.date.iloc[full].ge(threshold).to_numpy()])
                elif policy=='quarter_equal':
                    q = pd.to_datetime(obs.date.iloc[tr]).dt.to_period('Q')
                    mass = pd.DataFrame({'q':q.to_numpy(), 'w':weights}).groupby('q').w.sum().to_numpy()
                    np.testing.assert_allclose(mass, n/len(mass), rtol=0, atol=1e-10)
                for seed in cfg()['training']['seeds']:
                    rng = np.random.default_rng(seed)
                    for _ in range(20):
                        order = legacy.epoch_order(len(tr), n, rng)
                        self.assertEqual(len(order), n)
                        self.assertTrue(((order>=0)&(order<len(tr))).all())
                        counts = np.bincount(order, minlength=len(tr))
                        self.assertLessEqual(int(counts.max()-counts.min()), 1)
            self.assertTrue(obs.weekday.iloc[te].eq(4).all())
            self.assertTrue(obs.date.iloc[te].gt(fold['cutoff']).all())
            self.assertTrue(obs.date.iloc[te].le(fold['end']).all())
        jobs = fit_plan()
        self.assertEqual(len(jobs), 45)
        self.assertEqual([sum(j['worker']==w for j in jobs) for w in [0,1]], [23,22])
        self.assertFalse(any(j['policy']=='full' for j in jobs))

    def test_weighted_scales_loss_and_uniform_training_parity(self):
        cutoff = cfg()['folds'][0]['cutoff']
        tr, weights, _, _ = load_policy(cutoff, 'full')
        labels, scales = targets(tr, weights)
        old_labels, old_scales = legacy.standardized_targets(tr)
        self.assertEqual(scales, old_scales)
        for k in labels: self.assertTrue(torch.equal(labels[k], old_labels[k]))
        values = legacy.batch_tensors(tr)
        states = []; cpu_rngs = []; gpu_rngs = []
        for use_new in [False, True]:
            model = legacy.make_model('combined', 20260910)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
            model.train(); order = np.random.default_rng(20260910).permutation(len(tr))
            for lo in range(0, len(tr), 128):
                batch = torch.tensor(order[lo:lo+128], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                p, a = legacy.predict(model, {k:t[batch] for k,t in values.items()}, True)
                if use_new: loss = joint_loss(p, a, labels, batch, None)
                else: loss = ((p-old_labels['returns'][batch])**2).mean()+.1*((a-old_labels['auxiliary'][batch])**2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            states.append({k:t.detach().cpu().clone() for k,t in model.state_dict().items()})
            cpu_rngs.append(torch.get_rng_state()); gpu_rngs.append(torch.cuda.get_rng_state())
            del model, optimizer
        self.assertTrue(all(torch.equal(states[0][k], states[1][k]) for k in states[0]))
        self.assertTrue(torch.equal(cpu_rngs[0], cpu_rngs[1])); self.assertTrue(torch.equal(gpu_rngs[0], gpu_rngs[1]))
        tr, weights, _, _ = load_policy(cutoff, 'quarter_equal')
        labels, _ = targets(tr, weights)
        arrays, _ = target_arrays(tr, weights)
        for k in arrays:
            np.testing.assert_allclose(np.average(arrays[k], axis=0, weights=weights), 0., atol=1e-12)
            np.testing.assert_allclose(np.average(arrays[k]**2, axis=0, weights=weights), 1., atol=1e-12)
        batch = torch.tensor([0,100,500,1000,1677], device='cuda')
        p = torch.arange(5, dtype=torch.float32, device='cuda')/10
        a = torch.zeros((5,25), device='cuda')
        w = torch.tensor(weights, dtype=torch.float32, device='cuda')
        measured = joint_loss(p, a, labels, batch, w)
        manual = sum(w[b]*((p[i]-labels['returns'][b])**2+.1*((a[i]-labels['auxiliary'][b])**2).mean()) for i,b in enumerate(batch))/len(batch)
        torch.testing.assert_close(measured, manual, rtol=2e-7, atol=1e-7)
        del values, labels; torch.cuda.empty_cache()

    def test_ridge_weighted_objective(self):
        rng = np.random.default_rng(812)
        x = rng.normal(size=(31,7)); y = rng.normal(size=31); w = rng.uniform(.2,2,size=31)
        model = fit_ridge(x,y,w,100)
        z = (x-model['feature_mean'])/model['feature_sd']; mass = w*100/w.sum()
        matrix = np.vstack([np.sqrt(mass)[:,None]*z, np.sqrt(10)*np.eye(7)])
        target = np.r_[np.sqrt(mass)*(y-model['target_mean']), np.zeros(7)]
        beta = np.linalg.lstsq(matrix,target,rcond=None)[0]
        np.testing.assert_allclose(model['coefficients'], beta, rtol=0, atol=1e-12)
        self.assertLess(abs(np.dot(mass,y-model['target_mean']-z@beta)), 1e-10)


if __name__ == '__main__':
    legacy.initialize(); check_frozen(require_contract=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    save(OUT / 'contract_verification.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures), source_sha256=source_hashes(),
        completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    if not result.wasSuccessful(): raise SystemExit(1)
