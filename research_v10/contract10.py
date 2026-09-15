"""Pre-fit loss, intercept, data and actual-network parity checks."""
from common10 import *
import unittest


class Contracts(unittest.TestCase):
    def test_scaled_huber_formula_and_derivative(self):
        e=torch.tensor([-12.,-2.,-1.001,-1.,-.3,0.,.3,1.,1.001,2.,12.],dtype=torch.float64,requires_grad=True)
        actual=robust_loss(e)
        reference=2*torch.nn.functional.huber_loss(e,torch.zeros_like(e),reduction='none',delta=1.)
        torch.testing.assert_close(actual,reference,rtol=0,atol=1e-12)
        gradient=torch.autograd.grad(actual.sum(),e)[0]
        torch.testing.assert_close(gradient,2*e.detach().clamp(-1,1),rtol=0,atol=1e-12)
        self.assertLessEqual(float(gradient.abs().max()),2.)
        quadratic=e.detach().abs()<=1
        torch.testing.assert_close(actual.detach()[quadratic],e.detach()[quadratic]**2,rtol=0,atol=0)

    def test_intercept_and_causal_data_budget(self):
        self.assertAlmostEqual(huber_location([-4,-2,-.5,0,.5,2,4]),0.,places=12)
        self.assertAlmostEqual(huber_location([-.5,-.1,0,.1,.5,100]),.2,places=12)
        records=json.loads((OUT/'training_scales_and_intercepts.json').read_text(encoding='utf-8'))
        obs=pd.read_csv(OUT/'observation_table.csv');dates=[]
        with np.load(V5/'cache/targets.npz') as data:raw=data['returns'].copy()
        for fold in cfg()['folds']:
            tr,te=load_fold(fold['cutoff']);n=len(tr)
            np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(fold['cutoff']).to_numpy()))
            expected=np.flatnonzero((obs.date.gt(fold['cutoff'])&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
            np.testing.assert_array_equal(te,expected);dates+=obs.date.iloc[te].tolist()
            record=next(r for r in records if r['cutoff']==fold['cutoff']);z=(raw[tr]-raw[tr].mean())/raw[tr].std()
            c=record['huber_location_standardized'];self.assertLess(abs(np.mean(np.clip(c-z,-1,1))),1e-12)
            for direction in [-1,1]:self.assertGreater(np.mean(numpy_robust(c+direction*.01-z)),np.mean(numpy_robust(c-z)))
            self.assertAlmostEqual(record['huber_location_return'],raw[tr].mean()+raw[tr].std()*c,places=12)
            for seed in cfg()['training']['seeds']:
                rng=np.random.default_rng(seed);expected_rng=np.random.default_rng(seed)
                for _ in range(20):
                    order=legacy.epoch_order(n,n,rng);np.testing.assert_array_equal(order,expected_rng.permutation(n))
                    chunks=batches(order);np.testing.assert_array_equal(np.concatenate(chunks),order)
                    self.assertEqual(len(chunks),(n+127)//128);self.assertEqual(len(chunks[-1]),n%128 or 128)
                    np.testing.assert_array_equal(np.sort(order),np.arange(n))
        self.assertEqual(len(dates),141);self.assertEqual(len(set(dates)),141);self.assertEqual(len(fit_plan()),9)
        refs=json.loads((OUT/'mse_checkpoint_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(refs),9);self.assertTrue(all(r['arm']=='combined' and r['epoch']==20 for r in refs))

    def test_quadratic_region_actual_network_and_rng_parity(self):
        tr,_=load_fold(cfg()['folds'][0]['cutoff']);tr=tr[:256]
        values=legacy.batch_tensors(tr);labels,_=legacy.standardized_targets(tr)
        # Bounded synthetic labels exercise nonzero return-head learning entirely
        # in the common quadratic region; auxiliary targets remain nontrivial.
        labels['returns']=.05*torch.tanh(labels['returns'])
        states=[];cpu=[];gpu=[]
        for rule in ['mse','huber']:
            model=legacy.make_model('combined',20260910).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
            order=np.random.default_rng(20260910).permutation(len(tr))
            for positions in batches(order):
                b=torch.tensor(positions,device='cuda');optimizer.zero_grad(set_to_none=True)
                p,a=legacy.predict(model,{k:t[b] for k,t in values.items()},True)
                self.assertLess(float((p-labels['returns'][b]).detach().abs().max()),1.)
                loss=joint_loss(p,a,labels,b,rule);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            states.append({k:t.detach().cpu().clone() for k,t in model.state_dict().items()})
            cpu.append(torch.get_rng_state());gpu.append(torch.cuda.get_rng_state())
            del model,optimizer
        self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in states[0]))
        self.assertTrue(torch.equal(cpu[0],cpu[1]));self.assertTrue(torch.equal(gpu[0],gpu[1]))
        del values,labels;torch.cuda.empty_cache()


if __name__=='__main__':
    legacy.initialize();check_frozen(contract=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    save(OUT/'contract_verification.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'protocol.json'),
        installed_reference='2*torch.nn.functional.huber_loss(delta=1)',torch_version=torch.__version__,completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    if not result.wasSuccessful():raise SystemExit(1)
