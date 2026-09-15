"""Pre-fit invariants and nontrivial loss/gradient parity tests."""
from common9 import *
import unittest


class Contracts(unittest.TestCase):
    def test_maturity_coverage_order_steps_and_coefficients(self):
        obs=pd.read_csv(OUT/'observation_table.csv');all_dates=[]
        for fold in cfg()['folds']:
            tr,te=load_fold(fold['cutoff']);n=len(tr);steps=(n+127)//128
            np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(fold['cutoff']).to_numpy()))
            expected=np.flatnonzero((obs.date.gt(fold['cutoff'])&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
            np.testing.assert_array_equal(te,expected);all_dates+=obs.date.iloc[te].tolist()
            for seed in cfg()['training']['seeds']:
                rng=np.random.default_rng(seed);rng_old=np.random.default_rng(seed)
                for _ in range(20):
                    order=legacy.epoch_order(n,n,rng);np.testing.assert_array_equal(order,rng_old.permutation(n))
                    chunks=batches(order,'balanced');self.assertEqual(len(chunks),steps)
                    np.testing.assert_array_equal(np.concatenate(chunks),order)
                    np.testing.assert_array_equal(np.sort(np.concatenate(chunks)),np.arange(n))
                    sizes=np.array([len(b) for b in chunks]);self.assertLessEqual(sizes.max()-sizes.min(),1)
                    coeff=np.zeros(n)
                    for b in chunks:coeff[b]+=batch_factor(len(b),n,'balanced')/len(b)/steps
                    np.testing.assert_allclose(coeff,1/n,rtol=0,atol=1e-15)
                    self.assertEqual(len(batches(order,'legacy')),len(chunks))
            self.assertGreater(min(len(b) for b in batches(np.arange(n),'balanced')),100)
        self.assertEqual(len(all_dates),141);self.assertEqual(len(set(all_dates)),141)
        self.assertEqual(len(fit_plan()),9)
        refs=json.loads((OUT/'legacy_checkpoint_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(refs),9);self.assertTrue(all(r['arm']=='combined' and r['epoch']==20 for r in refs))

    def test_unequal_batch_full_objective_and_gradient_identity(self):
        torch.manual_seed(917);n=257
        p=torch.randn(n,dtype=torch.float64,requires_grad=True)
        a=torch.randn(n,25,dtype=torch.float64,requires_grad=True)
        labels={'returns':torch.randn(n,dtype=torch.float64),'auxiliary':torch.randn(n,25,dtype=torch.float64)}
        full=((p-labels['returns'])**2).mean()+.1*((a-labels['auxiliary'])**2).mean()
        order=np.random.default_rng(918).permutation(n)
        values={}
        for rule in ['legacy','balanced']:
            losses=[]
            for b in batches(order,rule):
                indices=torch.tensor(b)
                losses.append(batch_loss(p[indices],a[indices],labels,indices,n,rule))
            values[rule]=torch.stack(losses).mean()
        torch.testing.assert_close(values['balanced'],full,rtol=0,atol=1e-12)
        actual=torch.autograd.grad(values['balanced'],[p,a],retain_graph=True)
        reference=torch.autograd.grad(full,[p,a],retain_graph=True)
        for x,y in zip(actual,reference):torch.testing.assert_close(x,y,rtol=0,atol=1e-12)
        self.assertGreater(abs(float((values['legacy']-full).detach())),1e-5)

    def test_divisible_batch_legacy_tensor_and_rng_parity(self):
        tr,_=load_fold(cfg()['folds'][0]['cutoff']);tr=tr[:256]
        values=legacy.batch_tensors(tr);labels,_=legacy.standardized_targets(tr)
        states=[];rng_cpu=[];rng_gpu=[]
        for new in [False,True]:
            model=legacy.make_model('combined',20260910).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
            order=np.random.default_rng(20260910).permutation(len(tr))
            chunks=batches(order,'balanced' if new else 'legacy')
            for positions in chunks:
                b=torch.tensor(positions,device='cuda');optimizer.zero_grad(set_to_none=True)
                p,a=legacy.predict(model,{k:t[b] for k,t in values.items()},True)
                loss=batch_loss(p,a,labels,b,len(tr),'balanced') if new else (
                    ((p-labels['returns'][b])**2).mean()+.1*((a-labels['auxiliary'][b])**2).mean())
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            states.append({k:t.detach().cpu().clone() for k,t in model.state_dict().items()})
            rng_cpu.append(torch.get_rng_state());rng_gpu.append(torch.cuda.get_rng_state())
            del model,optimizer
        self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in states[0]))
        self.assertTrue(torch.equal(rng_cpu[0],rng_cpu[1]));self.assertTrue(torch.equal(rng_gpu[0],rng_gpu[1]))
        del values,labels;torch.cuda.empty_cache()


if __name__=='__main__':
    legacy.initialize();check_frozen(contract=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    save(OUT/'contract_verification.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),source_sha256=source_hashes(),
        protocol_sha256=sha(ROOT/'protocol.json'),completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    if not result.wasSuccessful():raise SystemExit(1)
