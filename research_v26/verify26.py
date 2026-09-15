from common26 import *
from contract26 import data_contract
import evaluate26 as evaluation
import verify15 as independent
from scipy.optimize import minimize
from scipy.special import expit
def independent_inputs(f,m,u,d):
    rep=(np.clip(f,d['rep__lower'],d['rep__upper'])-d['rep__mean'])/d['rep__sd'];mk=(np.clip(m,d['market__lower'],d['market__upper'])-d['market__mean'])/d['market__sd']
    a=np.column_stack([rep,mk]);v=mk[:,1]*(rep@d['vol__ray'])-a@d['vol__projection'][:-1]-d['vol__projection'][-1];v=(v-d['vol__residual_mean'])/d['vol__residual_sd'];b=np.column_stack([a,v])
    h=u*(rep@d['order__ray'])-b@d['order__projection'][:-1]-d['order__projection'][-1];h=(h-d['order__residual_mean'])/d['order__residual_sd'];c=np.column_stack([b,h]);return [a,b,c,c]
def main():
    legacy.initialize();start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();dcheck=data_contract();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for k in ['source_sha256','protocol_sha256','input_sha256']:assert run[k]==prep[k]
    obs,price,targets=data();models=read(OUT/'models.json');heads=read(OUT/'heads.json');testrefs=read(OUT/'testing_features.json');pred=csv('model_predictions');ensemble=csv('ensemble_predictions');curves=csv('training_curves');scales=read(OUT/'training_scales.json');solutions=[];replays=[];maxpred=0.;maxfeatures=0.
    for r in models:
        fold=next(f for f in cfg()['folds'] if f['cutoff']==r['cutoff']);tr,te=indices(obs,fold);hs=[next(h for h in heads if h['cutoff']==r['cutoff'] and h['seed']==r['seed'] and h['method']==m) for m in LEARNED];d=arrays(hs[0]);model,state=load_model(r)
        assert state['scales']==scales[r['cutoff']] and state['protocol_sha256']==prep['protocol_sha256'] and state['train_n']==len(tr)
        assert training.object_hash(state['optimizer_state_dict'])==r['optimizer_sha256']
        assert training.object_hash({k:state[k] for k in ['numpy_rng_state','cpu_rng_state','cuda_rng_state']})==r['rng_sha256']
        cs=curves[curves.cutoff.eq(r['cutoff'])&curves.seed.eq(r['seed'])].sort_values('epoch');assert len(cs)==20 and cs.epoch.tolist()==list(range(1,21));rng=np.random.default_rng(r['seed'])
        for row in cs.itertuples():
            assert row.order_sha256==training.array_hash(rng.permutation(len(tr))) and row.presentations==len(tr) and row.optimizer_steps==(len(tr)+127)//128
        assert rng.bit_generator.state==state['numpy_rng_state']
        for s in state['optimizer_state_dict']['state'].values():assert int(s['step'])==20*((len(tr)+127)//128)
        values=legacy.batch_tensors(tr);f,native=neural.extract_features(model,values);np.testing.assert_array_equal(f,d['features']);np.testing.assert_array_equal(native,d['native_standardized']);np.testing.assert_array_equal(tr,d['row_index'])
        for prefix,input_f in [('rep',f),('market',market(price,obs,tr))]:
            actual=fit_clip(input_f)
            for k in ['lower','upper','mean','sd','standardized']:np.testing.assert_allclose(actual[k],d[prefix+'__'+k],rtol=0,atol=1e-12)
        mf=market(price,obs,tr);u=gates(price,obs,tr);xx=independent_inputs(f,mf,u,d);y=(targets['returns'][tr]>0).astype(float);np.testing.assert_array_equal(y,d['direction'])
        for x,key in zip(xx[:3],['x18','x19','x23']):np.testing.assert_allclose(x,d[key],rtol=0,atol=1e-9)
        t18,t19,t23,t25=[np.asarray(h['coefficients']) for h in hs];np.testing.assert_array_equal(d['vol__ray'],t18[:25]);np.testing.assert_array_equal(d['order__ray'],t18[:25]);np.testing.assert_array_equal(np.r_[t25[:-2],t25[-1]],t19)
        for prefix,x,gate in [('vol',xx[0],d['market__standardized'][:,1]),('order',xx[1],u)]:
            product=gate*(x[:,:25]@t18[:25]);a=design(x);proj=np.linalg.lstsq(a,product,rcond=1e-12)[0];np.testing.assert_allclose(proj,d[prefix+'__projection'],rtol=1e-7,atol=1e-8)
            residual=product-a@proj;assert abs((a.T@d[prefix+'__interaction'])/len(tr)).max()<1e-8
            assert abs(residual.std()-d[prefix+'__residual_sd'])<1e-9
        for j,(h,x) in enumerate(zip(hs,xx)):
            theta=np.asarray(h['coefficients']);primary_p=expit(x@theta[:-1]+theta[-1])
            if j<3:
                init=np.zeros(x.shape[1]+1);rate=y.mean();init[-1]=np.log(rate/(1-rate))
                alt=minimize(lambda t:independent.independent_objective(t,x,y,.01),init,jac=True,method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=40));val,grad=independent.independent_objective(alt.x,x,y,.01)
                p=expit(x@alt.x[:-1]+alt.x[-1]);gap=float(abs(p-primary_p).max());objgap=abs(val-h['objective']);gn=float(abs(grad).max());assert objgap<=1e-10 and gap<=1e-5 and gn<=5e-7
                solutions.append(dict(job=h['job'],solver='L-BFGS-B',solver_success=bool(alt.success),iterations=int(alt.nit),probability_gap=gap,objective_gap=objgap,gradient_inf=gn))
            else:
                offset=xx[1]@t19[:-1]+t19[-1];alt=previous.independent_root(offset,x[:,-1],y,cfg()['offset_probe']);gap=abs(alt['gamma']-theta[-2]);assert gap<=1e-7
                assert abs(alt['objective']-h['reduced_objective'])<=1e-10
                solutions.append(dict(job=h['job'],solver='brentq',solver_success=alt['converged'],iterations=alt['iterations'],coefficient_gap=gap,objective_gap=abs(alt['objective']-h['reduced_objective']),gradient_inf=abs(alt['gradient'])))
        td=arrays(next(t for t in testrefs if t['cutoff']==r['cutoff'] and t['seed']==r['seed']));f,native=forecasts(model,te,scales[r['cutoff']]);np.testing.assert_array_equal(f,td['features']);np.testing.assert_array_equal(native,td['native']);np.testing.assert_array_equal(te,td['row_index'])
        xx=independent_inputs(f,market(price,obs,te),gates(price,obs,te),d)
        for x,key in zip(xx[:3],['x18','x19','x23']):
            gap=float(abs(x-td[key]).max());maxfeatures=max(maxfeatures,gap);assert gap<1e-9
        for h,x in zip(hs,xx):
            g=pred[pred.method.eq(h['method'])&pred.cutoff.eq(r['cutoff'])&pred.seed.eq(r['seed'])].sort_values('row_index');np.testing.assert_array_equal(te,g.row_index);theta=np.asarray(h['coefficients']);p=expit(x@theta[:-1]+theta[-1]);gap=float(abs(p-g.probability.to_numpy()).max());maxpred=max(maxpred,gap);assert gap<1e-10
            np.testing.assert_array_equal(g.direction_up,(p>.5).astype(int));np.testing.assert_array_equal(g.actual_up,(targets['returns'][te]>0).astype(int));np.testing.assert_allclose(g.actual,targets['returns'][te],rtol=0,atol=1e-14)
        g=pred[pred.method.eq('native_mse')&pred.cutoff.eq(r['cutoff'])&pred.seed.eq(r['seed'])].sort_values('row_index');np.testing.assert_allclose(g.score,native,rtol=0,atol=1e-14);np.testing.assert_array_equal(g.direction_up,(native>0).astype(int))
        g=pred[pred.method.eq('training_frequency')&pred.cutoff.eq(r['cutoff'])];np.testing.assert_allclose(g.probability,y.mean(),rtol=0,atol=1e-14)
        replays.append(dict(cutoff=r['cutoff'],seed=r['seed'],training_rows=len(tr),testing_rows=len(te),checkpoint_unchanged=True,training_features_bitwise_equal=True,testing_features_bitwise_equal=True,native_bitwise_equal=True))
        assert training.object_hash(model.state_dict())==r['model_sha256'];del model,values;torch.cuda.empty_cache();print(f'Verified {len(replays)}/18 backbones and {len(solutions)}/72 solvers',flush=True)
    actual=ensemble_from(pred);pd.testing.assert_frame_equal(actual,ensemble,check_dtype=False,atol=1e-14,rtol=0)
    tables,pairs=evaluation.compute(pred,ensemble)
    for name,table in tables.items():pd.testing.assert_frame_equal(table,csv(name),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'))
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['method','year']);independent.verify_metric_table(pred[pred.seed.ne(-1)],csv('seed_yearly_metrics'),['method','seed','year'])
    for w in cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end='2026-08-31',n=272)]:
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['method'])
        independent.verify_metric_table(evaluation.select(pred[pred.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['method','seed'])
    assert len(solutions)==72 and len(replays)==18;assert old_evidence()==prep['old_evidence'];check_frozen()
    pd.DataFrame(solutions).to_csv(OUT/'independent_solutions.csv',index=False);pd.DataFrame(replays).to_csv(OUT/'checkpoint_replays.csv',index=False)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=len(prep['old_evidence']),data_contract=dcheck,checkpoint_replays=18,independent_head_solutions=72,maximum_feature_algebra_gap=maxfeatures,maximum_forecast_gap=maxpred,training_feature_rows=54252,testing_feature_rows=816,independent_metrics=True,all_predefined_comparisons_recomputed=True,artifacts={n:sha(OUT/n) for n in ['independent_solutions.csv','checkpoint_replays.csv']}))
    print('Verification PASS: all models, convex heads, predictions, metrics and old evidence.',flush=True)
if __name__=='__main__':main()
