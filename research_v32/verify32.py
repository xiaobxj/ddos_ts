from common32 import *
from contract32 import memberships_check,boundary_cases,independent_route
import evaluate32 as evaluation
import verify15 as independent
from verify28 import independent_inputs
from scipy.optimize import minimize
from scipy.special import expit
import math

def verify_inference(ensemble,pairs):
    ps=[]
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(0,n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);np.testing.assert_array_equal(ids,statistics.bootstrap_indices(n))
        for row in [r for r in pairs if r['window']==w['name']]:
            e=ensemble[ensemble.date.between(w['start'],w['end'])];a=e[e.history.eq(row['history'])&e.method.eq(row['candidate'])].sort_values('date');b=e[e.history.eq(row['reference_history'])&e.method.eq(row['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==n
            if row['metric']=='direction_error':v=(a.direction_up.to_numpy()!=a.actual_up.to_numpy()).astype(float)-(b.direction_up.to_numpy()!=b.actual_up.to_numpy()).astype(float)
            else:v=(a.probability.to_numpy()-a.actual_up.to_numpy())**2-(b.probability.to_numpy()-b.actual_up.to_numpy())**2
            mu=float(v.mean());boot=v[ids].mean(1);center=(v-mu)[ids].mean(1);p=(1+int((np.abs(center)>=abs(mu)).sum()))/10001;lo,hi=np.quantile(boot,[.025,.975]);assert abs(mu-row['difference'])<1e-14 and abs(lo-row['ci95_low'])<1e-14 and abs(hi-row['ci95_high'])<1e-14 and p==row['p'];ps.append(p)
    assert ps==[r['p'] for r in pairs];last=0.;adjusted=[0.]*len(ps)
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):last=max(last,min(1.,ps[i]*(len(ps)-rank)));adjusted[i]=last
    np.testing.assert_array_equal(adjusted,[r['holm_adjusted_p'] for r in pairs])

def main():
    legacy.initialize();started=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['extraction','fitting','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    checks=memberships_check();assert len(boundary_cases())==6;obs,price,targets=data();allrows=np.arange(len(obs));mf=market(price,obs,allrows);u=gates(price,obs,allrows);y=(targets['returns']>0).astype(float);models=read(OUT/'source_models.json');banks=read(OUT/'feature_banks.json');heads=read(OUT/'heads.json');pred=csv('model_predictions');ensemble=csv('ensemble_predictions');candidate=pred[pred.history.eq(NEW[0])];route=csv('routing');scales=read(V28/'results/training_scales.json');replays=[];solutions=[];interfaces=[];maxpred=maxfeature=0.
    for br in banks:
        annual=br['cutoff'];seed=br['seed'];modelref=next(r for r in models if r['cutoff']==annual and r['seed']==seed);model,state=load_model(modelref);assert state['protocol_sha256']==sha(V28/'protocol.json') and state['history']=='rolling5' and state['scales']==scales['rolling5_'+annual];assert training.object_hash(state['optimizer_state_dict'])==modelref['optimizer_sha256'];bank=arrays(br)
        values=legacy.batch_tensors(bank['missing_rows']);f,z=neural.extract_features(model,values);np.testing.assert_array_equal(f,bank['missing_features']);np.testing.assert_array_equal(z,bank['missing_native_standardized'])
        a=arrays(br['annual_training_source']);b=arrays(br['annual_testing_source']);lookup={int(i):x for i,x in zip(a['row_index'],a['features'])};lookup.update({int(i):x for i,x in zip(b['row_index'],b['features'])});assert not set(bank['missing_rows'])&set(lookup);lookup.update({int(i):x for i,x in zip(bank['missing_rows'],f)});np.testing.assert_array_equal(np.stack([lookup[int(i)] for i in bank['row_index']]),bank['features'])
        assert training.object_hash(model.state_dict())==modelref['model_sha256'];replays.append(dict(encoder_cutoff=annual,seed=seed,new_rows=len(f),source_features_exact=True,new_features_exact=True,network_unchanged=True));del model,state,values;torch.cuda.empty_cache()
    for (cutoff,seed),group in pd.DataFrame(heads).groupby(['cutoff','seed'],sort=True):
        hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED];annual=annual_for(cutoff);bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));tr=training_rows(obs,cutoff);d=arrays(hs[0]);f,m,v,labels=training_interface(bank,tr,mf,u,y);np.testing.assert_array_equal(d['row_index'],tr)
        for key,value in [('features',f),('market_features',m),('gate',v),('direction',labels)]:np.testing.assert_array_equal(d[key],value)
        # Poison everything the fitting interface must not receive; no refit needed.
        badbank={k:value.copy() for k,value in bank.items()};outside=~np.isin(bank['row_index'],tr);badbank['features'][outside]=123456.;bad_y=y.copy();bad_y[obs.joint_completed.gt(cutoff).to_numpy()]=1-bad_y[obs.joint_completed.gt(cutoff).to_numpy()];poisoned=training_interface(badbank,tr,mf,u,bad_y)
        for value,bad in zip((f,m,v,labels),poisoned):np.testing.assert_array_equal(value,bad)
        interfaces.append(dict(head_cutoff=cutoff,encoder_cutoff=annual,seed=int(seed),n=len(tr),outside_feature_rows_poisoned=int(outside.sum()),future_labels_excluded=True,fit_inputs_unchanged=True))
        for prefix,x in [('rep',f),('market',m)]:
            low,high=np.quantile(np.asarray(x,float),[.01,.99],axis=0);clipped=np.clip(x,low,high);mean=clipped.mean(0);sd=np.maximum(clipped.std(0),1e-6)
            for key,value in [('lower',low),('upper',high),('mean',mean),('sd',sd),('standardized',(clipped-mean)/sd)]:np.testing.assert_allclose(value,d[prefix+'__'+key],rtol=0,atol=1e-12)
        xx=independent_inputs(f,m,v,d);t18,t19,t23,t25=[np.asarray(h['coefficients']) for h in hs];np.testing.assert_array_equal(d['vol__ray'],t18[:25]);np.testing.assert_array_equal(d['order__ray'],t18[:25]);np.testing.assert_array_equal(np.r_[t25[:-2],t25[-1]],t19)
        for x,key in zip(xx[:3],['x18','x19','x23']):np.testing.assert_allclose(x,d[key],rtol=0,atol=1e-9)
        for prefix,x,gate in [('vol',xx[0],d['market__standardized'][:,1]),('order',xx[1],v)]:
            product=gate*(x[:,:25]@t18[:25]);a=design(x);proj=np.linalg.lstsq(a,product,rcond=1e-12)[0];np.testing.assert_allclose(proj,d[prefix+'__projection'],rtol=1e-7,atol=1e-8);residual=product-a@proj;assert abs((a.T@d[prefix+'__interaction'])/len(tr)).max()<1e-8 and abs(residual.std()-d[prefix+'__residual_sd'])<1e-9
        for j,(h,x) in enumerate(zip(hs,xx)):
            theta=np.asarray(h['coefficients']);p=expit(x@theta[:-1]+theta[-1])
            if j<3:
                initial=np.zeros(x.shape[1]+1);rate=labels.mean();initial[-1]=np.log(rate/(1-rate));alt=minimize(lambda t:independent.independent_objective(t,x,labels,.01),initial,jac=True,method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=40));val,grad=independent.independent_objective(alt.x,x,labels,.01);gap=float(abs(expit(x@alt.x[:-1]+alt.x[-1])-p).max());objgap=abs(val-h['objective']);gn=float(abs(grad).max());assert objgap<=1e-10 and gap<=1e-5 and gn<=5e-7;solutions.append(dict(job=h['job'],solver='L-BFGS-B',solver_success=bool(alt.success),iterations=int(alt.nit),probability_gap=gap,objective_gap=objgap,gradient_inf=gn))
            else:
                offset=xx[1]@t19[:-1]+t19[-1];alt=prior.previous.independent_root(offset,x[:,-1],labels,cfg()['offset_probe']);gap=abs(alt['gamma']-theta[-2]);assert gap<=1e-7 and abs(alt['objective']-h['reduced_objective'])<=1e-10;solutions.append(dict(job=h['job'],solver='brentq',solver_success=alt['converged'],iterations=alt['iterations'],coefficient_gap=gap,objective_gap=abs(alt['objective']-h['reduced_objective']),gradient_inf=abs(alt['gradient'])))
        te=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);f=bank_slice(bank,te);xx=independent_inputs(f,mf[te],u[te],d);direct=apply_pipeline(f,mf[te],u[te],d)
        for h,x,direct_x in zip(hs,xx,direct):
            gap=float(abs(x-direct_x).max());maxfeature=max(maxfeature,gap);assert gap<1e-9;g=candidate[candidate.cutoff.eq(cutoff)&candidate.seed.eq(seed)&candidate.method.eq(h['method'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,te);theta=np.asarray(h['coefficients']);p=expit(x@theta[:-1]+theta[-1]);gap=float(abs(p-g.probability.to_numpy()).max());maxpred=max(maxpred,gap);assert gap<1e-10;np.testing.assert_array_equal(g.direction_up,(p>.5).astype(int))
        print(f'Verified {len(solutions)}/204 new convex solutions, {cutoff} seed {seed}.',flush=True)
    assert len(solutions)==204 and len(interfaces)==51 and len(replays)==18;np.testing.assert_allclose(candidate.actual,targets['returns'][candidate.row_index],rtol=0,atol=1e-14)
    base,be=baselines();pd.testing.assert_frame_equal(pred[~pred.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    key=['method','seed','date']
    for method,history in [('native_mse','rolling5_annual20'),('training_frequency','rolling5_quarterly20')]:
        a=candidate[candidate.method.eq(method)].drop(columns='history').sort_values(key).reset_index(drop=True);b=base[base.history.eq(history)&base.method.eq(method)].drop(columns='history').sort_values(key).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
    q1=route[route.head_cutoff.str.endswith('12-31')].row_index
    a=candidate[candidate.row_index.isin(q1)].drop(columns='history').sort_values(key).reset_index(drop=True);b=base[base.history.eq('rolling5_annual20')&base.row_index.isin(q1)].drop(columns='history').sort_values(key).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
    for r in candidate.itertuples():
        annual,head=independent_route(r.date);assert r.cutoff==(annual if r.method=='native_mse' else head)
    actual,sources=candidate_predictions(obs,heads,banks,route,base);pd.testing.assert_frame_equal(actual,candidate.reset_index(drop=True),check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(sources,csv('prediction_sources'),check_dtype=False);pd.testing.assert_frame_equal(ensemble_from(pred),ensemble,check_dtype=False,atol=1e-14,rtol=0)
    tables,pairs=evaluation.compute(pred,ensemble)
    for name,t in tables.items():pd.testing.assert_frame_equal(t,csv(name),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));assert len(pairs)==28;verify_inference(ensemble,pairs)
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['history','method','year']);independent.verify_metric_table(pred[pred.seed.ne(-1)],csv('seed_yearly_metrics'),['history','method','seed','year'])
    for w in evaluation.windows():
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['history','method']);independent.verify_metric_table(evaluation.select(pred[pred.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['history','method','seed'])
    assert len(pred)==34816 and len(ensemble)==13056;assert old_evidence()==prep['old_evidence'];check_frozen();files=[]
    for name,rows in [('feature_replays',replays),('independent_solutions',solutions),('training_interface_checks',interfaces)]:p=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(p,index=False);files.append(p)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=4759,annual_network_replays=18,new_neural_fits=0,new_head_fits=204,independent_head_solutions=204,causal_training_interfaces=51,membership_checks=len(checks),maximum_probability_gap=maxpred,maximum_feature_algebra_gap=maxfeature,all_q1_predictions_exact=True,native_exactly_annual=True,frequency_exactly_quarterly=True,all28comparisons_recomputed=True,independent_bootstrap_and_holm=True,independent_metrics=True,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Verification PASS: frozen annual features, 204 independent head solutions, causal inputs and all metrics.',flush=True)
if __name__=='__main__':main()
