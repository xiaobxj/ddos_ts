from common34 import *
from contract34 import memberships_check,boundary_cases,coefficient_cases,independent_route
import evaluate34 as evaluation
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
    for phase in ['fitting','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    checks=memberships_check();assert len(boundary_cases()+coefficient_cases())==13
    obs,price,targets=data();ids=np.arange(len(obs));mf=market(price,obs,ids);u=gates(price,obs,ids);y=(targets['returns']>0).astype(float)
    banks=read(OUT/'feature_banks.json');annuals=read(OUT/'source_heads.json');endpoints=read(OUT/'endpoints.json');policies=read(OUT/'policy_heads.json');training_refs=read(OUT/'training_designs.json');models=csv('model_predictions');ensemble=csv('ensemble_predictions');base,be=baselines();route=csv('routing');solutions=[];interfaces=[];drifts=[];maxfeature=maxpred=maxanchor=maxlogit=0.
    for trref in training_refs:
        cutoff=trref['cutoff'];annual=trref['encoder_cutoff'];seed=trref['seed'];rows=training_rows(obs,cutoff)
        ah=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED];eh=[next(h for h in endpoints if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED]
        bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));d=arrays(ah[0]);saved=arrays(trref);f=bank_slice(bank,rows);xx=independent_inputs(f,mf[rows],u[rows],d);labels=y[rows]
        np.testing.assert_array_equal(saved['row_index'],rows);np.testing.assert_array_equal(saved['direction'],labels)
        for x,key in zip(xx,['x18','x19','x23','x23']):
            gap=float(abs(x-saved[key]).max());maxfeature=max(maxfeature,gap);assert gap<1e-9
        # Poison every excluded row before passing data to the actual fit interface.
        badbank={k:v.copy() for k,v in bank.items()};outside=~np.isin(bank['row_index'],rows);badbank['features'][outside]=123456.
        out=~np.isin(ids,rows);bad_m=mf.copy();bad_m[out]=-98765.;bad_u=u.copy();bad_u[out]=42.;bad_y=y.copy();bad_y[out]=1-bad_y[out]
        normal_x,normal_y=training_interface(bank,rows,mf,u,y,d);poison_x,poison_y=training_interface(badbank,rows,bad_m,bad_u,bad_y,d)
        for a,b in zip(normal_x,poison_x):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(normal_y,poison_y);interfaces.append(dict(cutoff=cutoff,encoder_cutoff=annual,seed=seed,n=len(rows),excluded_bank_rows=int(outside.sum()),excluded_labels=int(out.sum()),all_inputs_unchanged=True))
        ts=[np.asarray(h['coefficients']) for h in eh];ta=[np.asarray(h['coefficients']) for h in ah]
        np.testing.assert_array_equal(np.r_[ts[3][:-2],ts[3][-1]],ts[1]);np.testing.assert_array_equal(np.r_[ta[3][:-2],ta[3][-1]],ta[1])
        for j,(h,x,t) in enumerate(zip(eh,xx,ts)):
            assert h['cache_file']==ah[j]['cache_file'] and h['cache_sha256']==ah[j]['cache_sha256'] and h['transform_cutoff']==annual
            if j<3:
                initial=np.zeros(x.shape[1]+1);rate=labels.mean();initial[-1]=np.log(rate/(1-rate));alt=minimize(lambda t:independent.independent_objective(t,x,labels,.01),initial,jac=True,method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=40));val,gr=independent.independent_objective(alt.x,x,labels,.01)
                gap=float(abs(expit(x@alt.x[:-1]+alt.x[-1])-expit(x@t[:-1]+t[-1])).max());objgap=abs(val-h['objective']);gn=float(abs(gr).max());assert objgap<=1e-10 and gap<=1e-5 and gn<=5e-7
                pv,pg=independent.independent_objective(t,x,labels,.01);assert abs(pv-h['objective'])<1e-12 and abs(pg).max()<2e-9
                solutions.append(dict(job=h['job'],solver='L-BFGS-B',solver_success=bool(alt.success),iterations=int(alt.nit),probability_gap=gap,objective_gap=objgap,gradient_inf=gn))
            else:
                offset=xx[1]@ts[1][:-1]+ts[1][-1];alt=prior.previous.independent_root(offset,x[:,-1],labels,cfg()['offset_probe']);gap=abs(alt['gamma']-t[-2]);assert gap<=1e-7 and abs(alt['objective']-h['reduced_objective'])<=1e-10
                solutions.append(dict(job=h['job'],solver='brentq',solver_success=alt['converged'],iterations=alt['iterations'],coefficient_gap=gap,objective_gap=abs(alt['objective']-h['reduced_objective']),gradient_inf=abs(alt['gradient'])))
        te=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);tx=independent_inputs(bank_slice(bank,te),mf[te],u[te],d)
        for policy in cfg()['updates']:
            ph=[next(h for h in policies if h['history']==policy['history'] and h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED];tp=[np.asarray(h['coefficients']) for h in ph];np.testing.assert_array_equal(np.r_[tp[3][:-2],tp[3][-1]],tp[1])
            for j,(h,x,a,b,t) in enumerate(zip(ph,tx,ta,ts,tp)):
                alpha=policy['alpha'];np.testing.assert_allclose(t,(1-alpha)*a+alpha*b,rtol=0,atol=1e-15)
                assert h['cache_file']==ah[j]['cache_file'] and h['cache_sha256']==ah[j]['cache_sha256'] and h['transform_cutoff']==annual
                old_z=x@a[:-1]+a[-1];new_z=x@b[:-1]+b[-1];z=np.array([math.fsum(float(v*w) for v,w in zip(np.r_[row,1.],t)) for row in x]);zgap=float(abs(z-((1-alpha)*old_z+alpha*new_z)).max());maxlogit=max(maxlogit,zgap);assert zgap<1e-11
                p=expit(z);g=models[models.history.eq(h['history'])&models.cutoff.eq(cutoff)&models.method.eq(h['method'])&models.seed.eq(seed)].sort_values('row_index');np.testing.assert_array_equal(g.row_index,te);gap=float(abs(p-g.probability.to_numpy()).max());maxpred=max(maxpred,gap);assert gap<1e-10;np.testing.assert_array_equal(g.direction_up,(p>.5).astype(int))
                annualg=base[base.history.eq('rolling5_annual20')&base.method.eq(h['method'])&base.seed.eq(seed)&base.row_index.isin(te)].sort_values('row_index');agap=float(abs(expit(old_z)-annualg.probability.to_numpy()).max());maxanchor=max(maxanchor,agap);assert agap<1e-10
                endpoint_norm=float(np.sqrt(sum(float(v*v) for v in b-a)));policy_norm=float(np.sqrt(sum(float(v*v) for v in t-a)));assert abs(policy_norm-alpha*endpoint_norm)<1e-14
                drifts.append(dict(history=h['history'],method=h['method'],cutoff=cutoff,seed=seed,alpha=alpha,annual_norm=float(np.linalg.norm(a)),endpoint_drift=endpoint_norm,policy_drift=policy_norm,intercept_change=float(t[-1]-a[-1]),maximum_abs_change=float(abs(t-a).max())))
        print(f'Verified {len(solutions)}/204 independent endpoints; {cutoff} seed {seed}.',flush=True)
    assert len(solutions)==204 and len(interfaces)==51 and len(drifts)==612
    key=['history','method','cutoff','seed'];pd.testing.assert_frame_equal(pd.DataFrame(drifts).sort_values(key).reset_index(drop=True),csv('coefficient_drift').sort_values(key).reset_index(drop=True),check_dtype=False,atol=1e-14,rtol=0)
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    key=['method','seed','date'];q1=route[route.head_cutoff.str.endswith('12-31')].row_index
    for history in NEW:
        g=models[models.history.eq(history)]
        for method,source in [('native_mse','rolling5_annual20'),('training_frequency','rolling5_quarterly20')]:
            a=g[g.method.eq(method)].drop(columns='history').sort_values(key).reset_index(drop=True);b=base[base.history.eq(source)&base.method.eq(method)].drop(columns='history').sort_values(key).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
        a=g[g.row_index.isin(q1)].drop(columns='history').sort_values(key).reset_index(drop=True);b=base[base.history.eq('rolling5_annual20')&base.row_index.isin(q1)].drop(columns='history').sort_values(key).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
        for r in g.itertuples():
            annual,head=independent_route(r.date);assert r.cutoff==(annual if r.method=='native_mse' else head)
    np.testing.assert_allclose(models.actual,targets['returns'][models.row_index],rtol=0,atol=1e-14);np.testing.assert_array_equal(models.actual_up,(models.actual>0).astype(int))
    indexed=ensemble.set_index(['history','method','date'])
    for key,g in models.groupby(['history','method','date']):
        r=indexed.loc[key];assert len(g)==(1 if key[1]=='training_frequency' else 3)
        score=math.fsum(g.score)/len(g);assert abs(score-r.score)<1e-14
        if key[1]=='native_mse':assert pd.isna(r.probability) and int(score>0)==r.direction_up
        else:
            p=math.fsum(g.probability)/len(g);assert abs(p-r.probability)<1e-14 and int(p>.5)==r.direction_up
    actual,sources=candidate_predictions(obs,policies,banks,route,base);pd.testing.assert_frame_equal(actual,models[models.history.isin(NEW)].reset_index(drop=True),check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(sources,csv('prediction_sources'),check_dtype=False)
    tables,pairs=evaluation.compute(models,ensemble)
    for name,t in tables.items():pd.testing.assert_frame_equal(t,csv(name),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));assert len(pairs)==60;verify_inference(ensemble,pairs)
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['history','method','year']);independent.verify_metric_table(models[models.seed.ne(-1)],csv('seed_yearly_metrics'),['history','method','seed','year'])
    for w in evaluation.windows():
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['history','method']);independent.verify_metric_table(evaluation.select(models[models.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['history','method','seed'])
    assert len(models)==47872 and len(ensemble)==17952;assert old_evidence()==prep['old_evidence'];check_frozen();files=[]
    for n,v in [('independent_solutions',solutions),('training_interface_checks',interfaces)]:p=OUT/f'{n}.csv';pd.DataFrame(v).to_csv(p,index=False);files.append(p)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=4946,new_neural_fits=0,new_feature_inference=0,new_transform_fits=0,independent_endpoint_solutions=204,derived_policy_head_checks=612,causal_training_interfaces=51,membership_checks=len(checks),maximum_probability_gap=maxpred,maximum_annual_anchor_probability_gap=maxanchor,maximum_feature_algebra_gap=maxfeature,maximum_logit_interpolation_gap=maxlogit,all_q1_predictions_exact=True,native_exactly_annual=True,frequency_exactly_quarterly=True,all60comparisons_recomputed=True,independent_bootstrap_and_holm=True,independent_metrics=True,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('Verification PASS: frozen annual transforms, independent endpoints, bounded steps, causal interfaces, metrics and all 60 contrasts.',flush=True)
if __name__=='__main__':main()
