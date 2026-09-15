from pathlib import Path
root=Path('D:/ddos_v3');old=(root/'research_v46/verify46.py').read_text(encoding='utf-8')
helpers=old[old.index('def eq('):old.index('def check_fits_and_validation():')]
stats=old[old.index('def check_statistics():'):old.index('\ndef main():')].replace('72','24')
code='''from common47 import *
from contract47 import independent_inputs,audit_sources
from scipy.optimize import minimize,brentq
from scipy.special import expit
import math

'''+helpers+'''
def independent_objective(theta,x,y,w):
    z=x@theta[:-1]+theta[-1];p=expit(z)
    value=math.fsum(float(a)*(float(np.logaddexp(0.,-b)) if c else float(np.logaddexp(0.,b))) for a,b,c in zip(w,z,y))+.005*float(theta[:-1]@theta[:-1])
    error=w*(p-y);gradient=np.r_[x.T@error+.01*theta[:-1],math.fsum(error)]
    return value,gradient

def check_fits():
    heads=read(OUT/'heads.json');assert len(heads)==72;solutions=[];maxgap=0.;obs=csv('observation_table');weights=csv('training_weights')
    for cutoff in cfg()['decision_dates']:
        for seed in cfg()['seeds']:
            hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in METHODS];d=arrays(hs[0]);src=arrays(dict(cache_file=hs[0]['frozen_pipeline_file'],cache_sha256=hs[0]['frozen_pipeline_sha256']))
            for key in ['row_index','x18','x19','x23','direction']:np.testing.assert_array_equal(d[key],src[key])
            z=weights[weights.cutoff.eq(cutoff)];np.testing.assert_array_equal(z.row_index,d['row_index']);np.testing.assert_array_equal(z.weight,d['weight']);y=d['direction'];w=d['weight'];assert abs(w.sum()-1)<1e-14
            for j,h in enumerate(hs):
                x=d[['x18','x19','x23','x23'][j]];theta=np.array(h['coefficients']);p=expit(x@theta[:-1]+theta[-1])
                assert h['train_n']==len(y) and h['iterations']<=100 and h['gradient_inf']<=(1e-9 if j<3 else 1e-10)
                if j<3:
                    initial=np.zeros(x.shape[1]+1);rate=math.fsum(w*y);initial[-1]=math.log(rate/(1-rate))
                    alt=minimize(lambda t:independent_objective(t,x,y,w),initial,jac=True,method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=40));value,gradient=independent_objective(alt.x,x,y,w);gap=float(abs(expit(x@alt.x[:-1]+alt.x[-1])-p).max());objgap=abs(value-h['objective']);gn=float(abs(gradient).max())
                    assert gap<=1e-5 and objgap<=1e-10 and gn<=5e-7,(h['job'],gap,objgap,gn)
                    actual_value,actual_grad=independent_objective(theta,x,y,w);eq(actual_value,h['objective']);assert abs(actual_grad).max()<=1.01e-9
                    solver='L-BFGS-B';iterations=int(alt.nit);success=bool(alt.success)
                else:
                    t19=np.array(hs[1]['coefficients']);np.testing.assert_array_equal(np.r_[theta[:-2],theta[-1]],t19);offset=d['x19']@t19[:-1]+t19[-1];feature=x[:,-1]
                    def grad(g):return math.fsum(w*feature*(expit(offset+g*feature)-y))+.01*g
                    bound=math.fsum(w*abs(feature))/.01+1;gamma,info=brentq(grad,-bound,bound,xtol=1e-13,rtol=1e-14,maxiter=200,full_output=True);gap=abs(gamma-theta[-2]);assert gap<1e-7
                    zz=offset+gamma*feature;value=math.fsum(float(a)*(float(np.logaddexp(0.,-b)) if c else float(np.logaddexp(0.,b))) for a,b,c in zip(w,zz,y))+.005*gamma*gamma
                    objgap=abs(value-h['reduced_objective']);gn=abs(grad(gamma));assert objgap<1e-10 and gn<1e-10;eq(h['objective'],h['reduced_objective']+.005*(t19[:-1]@t19[:-1]));assert abs(grad(theta[-2]))<=1.01e-10
                    solver='Brent';iterations=info.iterations;success=bool(info.converged)
                solutions.append(dict(job=h['job'],solver=solver,solver_success=success,iterations=iterations,solution_gap=gap,objective_gap=objgap,gradient_inf=gn));maxgap=max(maxgap,gap)
            assert hs[2]['objective']<=hs[3]['objective']+1e-12<=hs[1]['objective']+2e-12 and hs[1]['objective']<=hs[0]['objective']+1e-12
    assert len(solutions)==72
    p=OUT/'independent_solutions.csv';pd.DataFrame(solutions).to_csv(p,index=False)
    return dict(independent_joint_solutions=54,independent_scalar_solutions=18,maximum_independent_solution_gap=maxgap),[p]

def check_predictions():
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');heads=read(OUT/'heads.json');obs=csv('observation_table');source=read(OUT/'source_heads.json');new=models[models.history.isin(NEW)];maxgap=0.;rows=0
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_exact=True,check_dtype=False);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_exact=True,check_dtype=False)
    assert len(models)==121856 and len(ensemble)==45696 and len(new)==4352 and ensemble.history.nunique()==28
    assert not models.duplicated(['history','method','seed','row_index']).any() and not ensemble.duplicated(['history','method','row_index']).any()
    for ref in read(OUT/'source_testing_features.json'):
        d=arrays(ref);hs=[next(h for h in heads if h['cutoff']==ref['cutoff'] and h['seed']==ref['seed'] and h['method']==m) for m in METHODS];src=arrays(next(h for h in source if h['cutoff']==ref['cutoff'] and h['seed']==ref['seed']));xx=independent_inputs(d['features'],d['market_features'],d['gate'],src)
        for h,x in zip(hs,xx+[xx[-1]]):
            a=new[new.method.eq(h['method'])&new.cutoff.eq(ref['cutoff'])&new.seed.eq(ref['seed'])].sort_values('row_index');np.testing.assert_array_equal(a.row_index,d['row_index']);theta=np.array(h['coefficients']);p=expit(x@theta[:-1]+theta[-1]);gap=float(abs(p-a.probability).max());assert gap<1e-10;maxgap=max(maxgap,gap);np.testing.assert_array_equal(a.direction_up,p>.5);np.testing.assert_array_equal(a.probability,a.score);rows+=len(a)
            for r in a.itertuples():
                assert r.cutoff==f'{int(r.date[:4])-1}-12-31'<r.date and r.joint_completed==obs.joint_completed.iloc[r.row_index] and r.actual==obs.exec_return.iloc[r.row_index] and r.actual_up==int(r.actual>0)
    assert rows==3264
    for method in ['native_mse','training_frequency']:
        a=new[new.method.eq(method)].drop(columns='history').reset_index(drop=True);b=base[base.history.eq(ANNUAL)&base.method.eq(method)].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True,check_dtype=False)
    ei=ensemble.set_index(['history','method','row_index'])
    for (history,method,idx),g in models.groupby(['history','method','row_index'],sort=False):
        n=1 if method=='training_frequency' else 3;assert len(g)==n;r=ei.loc[(history,method,idx)];eq(r.score,math.fsum(g.score)/n)
        if method=='native_mse':assert pd.isna(r.probability) and r.direction_up==int(r.score>0)
        else:eq(r.probability,math.fsum(g.probability)/n);assert r.direction_up==int(r.probability>.5)
        if history in NEW and method not in METHODS:
            annual=ei.loc[(ANNUAL,method,idx)];assert r.score==annual.score and r.direction_up==annual.direction_up
            if method=='training_frequency':assert r.probability==annual.probability
    summary=csv('weight_summary').set_index('cutoff');freq=csv('weighted_frequency_predictions');assert len(freq)==272
    for r in freq.itertuples():assert r.cutoff==f'{int(r.date[:4])-1}-12-31' and r.probability==summary.loc[r.cutoff,'weighted_up_frequency'] and r.direction_up==int(r.probability>.5) and r.actual==obs.exec_return.iloc[r.row_index]
    prov=csv('prediction_sources');assert len(prov)==72
    for r in prov.itertuples():
        h=next(h for h in heads if h['job']==r.head_job);ref=next(t for t in read(OUT/'source_testing_features.json') if t['cutoff']==r.cutoff and t['seed']==r.seed)
        assert r.training_source==h['frozen_pipeline_file'] and r.testing_source==ref['cache_file'] and r.method==h['method'] and r.n==len(arrays(ref)['row_index'])
    joined=ensemble.merge(csv('weekly_context')[['row_index','date','state']],on=['row_index','date'],validate='many_to_one');count=0
    for name,source,keys in [('ensemble_metrics',ensemble,['history','method']),('seed_metrics',models,['history','method','seed']),('state_metrics',joined,['history','method','state'])]:
        groups={k:g for k,g in source.groupby(keys,sort=False)}
        for row in csv(name).to_dict('records'):
            w=next(w for w in periods() if w['name']==row['period']);g=groups.get(tuple(row[k] for k in keys),source.iloc[:0]);g=g[g.date.between(w['start'],w['end'])]
            for k,x in independent_metric(g).items():eq(x,row[k])
            count+=1
    for row in csv('weighted_frequency_metrics').to_dict('records'):
        w=next(w for w in periods() if w['name']==row['period']);g=freq[freq.date.between(w['start'],w['end'])]
        for k,x in independent_metric(g).items():eq(x,row[k])
        count+=1
    return dict(independent_learned_seed_predictions=rows,maximum_probability_gap=maxgap,independent_metric_cells=count,exact_native_frequency_controls=True,old_histories_preserved=27)

def check_diagnostics():
    weekly=csv('weekly_policy_effects');assert len(weekly)==1088;ensemble=csv('ensemble_predictions').set_index(['history','method','row_index'])
    for r in weekly.itertuples():
        a=ensemble.loc[(r.history,r.method,r.row_index)];assert a.probability==r.probability and a.actual_up==r.actual_up
        for label,history in [('annual',ANNUAL),('quarter4',QUARTER)]:
            b=ensemble.loc[(history,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;case='recovery' if good and not old else 'regression' if old and not good else 'stable_correct' if good else 'stable_wrong'
            assert getattr(r,label+'_probability')==b.probability and getattr(r,'case_vs_'+label)==case and getattr(r,'changed_vs_'+label)==(a.probability!=b.probability);eq(getattr(r,'brier_vs_'+label),(a.probability-a.actual_up)**2-(b.probability-b.actual_up)**2)
    for r in csv('reference_comparisons').itertuples():
        w=next(w for w in periods() if w['name']==r.period);g=weekly[weekly.method.eq(r.method)&weekly.date.between(w['start'],w['end'])];label='annual' if r.reference_history==ANNUAL else 'quarter4';assert r.n==len(g) and r.changed_probability_weeks==sum(g['changed_vs_'+label]) and r.recoveries==sum(g['case_vs_'+label]=='recovery') and r.regressions==sum(g['case_vs_'+label]=='regression');eq(r.brier_difference,math.fsum(g['brier_vs_'+label])/len(g))
    same_tables(csv('all_2026_cases'),weekly[weekly.year.eq(2026)])
    return dict(independent_weekly_effects=1088)
'''+stats+'''
def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();started=now();last=read(OUT/'contract_verification.json')['completed_utc']
    for phase in ['fitting','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    sources,gap=audit_sources();fit,files=check_fits();print('Independent72weightedoptima,frozenfeatures,equalweightreplays andcausalinterfaces PASS.',flush=True)
    pred=check_predictions();diag=check_diagnostics();check_statistics();assert old_evidence()==prep['old_evidence'];check_frozen()
    save(OUT/'verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=6056,new_neural_fits=0,new_feature_inference=0,new_head_fits=72,annual_source_jobs=18,uniform_head_replays=72,future_poison_interfaces=18,maximum_frozen_design_gap=gap,independent_comparisons=24,**fit,**pred,**diag,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('R47 independentpredictions,metrics,diagnostics and24comparisons PASS.',flush=True)
if __name__=='__main__':main()
'''
(root/'research_v47/verify47.py').write_text(code,encoding='utf-8')
print('R47 verification source staged.')
