from pathlib import Path
P=Path('D:/ddos_v3');old=(P/'research_v47/verify47.py').read_text(encoding='utf-8')
helpers=old[old.index('def eq('):old.index('def independent_objective(')]
stats=old[old.index('def check_statistics():'):old.index('\ndef main():')].replace('24','72')
code='''from common48 import *
from contract48 import audit_sources,independent_inputs
from scipy.special import expit
import math

'''+helpers+'''
def check_assembly():
    uniform=read(OUT/'uniform_heads.json');weighted=read(OUT/'weighted_heads.json');heads=read(OUT/'heads.json');table=csv('coefficient_sources');assert len(heads)==144 and len(table)==4500
    ui={(h['cutoff'],h['seed'],h['method']):h for h in uniform};wi={(h['cutoff'],h['seed'],h['method']):h for h in weighted};expected={(history,*k) for history in NEW for k in ui};assert {(h['history'],h['cutoff'],h['seed'],h['method']) for h in heads}==expected
    for h in heads:
        key=(h['cutoff'],h['seed'],h['method']);u=ui[key];w=wi[key];assert h['uniform_job']==u['job'] and h['weighted_job']==w['job'] and h['frozen_pipeline_file']==u['cache_file'] and h['frozen_pipeline_sha256']==u['cache_sha256'] and h['train_n']==u['train_n'] and h['model_project_file']==u['model_project_file']
        q=table[table.history.eq(h['history'])&table.cutoff.eq(h['cutoff'])&table.seed.eq(h['seed'])&table.method.eq(h['method'])].sort_values('coordinate');assert q.coordinate.tolist()==list(range(len(u['coefficients'])))
        for j,value in enumerate(h['coefficients']):
            intercept=j==len(h['coefficients'])-1;source='weighted' if (intercept and h['history']==INTERCEPT) or (not intercept and h['history']==SLOPES) else 'uniform';ref=w if source=='weighted' else u
            assert value==ref['coefficients'][j] and q.iloc[j]['value']==value and q.iloc[j]['source']==source and q.iloc[j]['role']==('intercept' if intercept else 'slope')
    return dict(assembled_heads_verified=144,coefficient_coordinates_verified=4500)

def check_predictions():
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');heads=read(OUT/'heads.json');obs=csv('observation_table');uniform=read(OUT/'uniform_heads.json');weighted=read(OUT/'weighted_heads.json');terms=csv('seed_logit_decomposition');maxgap=0.;maxidentity=0.;count=0
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_exact=True,check_dtype=False);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_exact=True,check_dtype=False)
    assert len(models)==130560 and len(ensemble)==48960 and ensemble.history.nunique()==30 and len(terms)==3264
    assert not models.duplicated(['history','method','seed','row_index']).any() and not ensemble.duplicated(['history','method','row_index']).any()
    mi=models.set_index(['history','method','seed','row_index']);ei=ensemble.set_index(['history','method','row_index']);ti=terms.set_index(['cutoff','method','seed','row_index']);ui={(h['cutoff'],h['seed'],h['method']):h for h in uniform};wi={(h['cutoff'],h['seed'],h['method']):h for h in weighted};hi={(h['history'],h['cutoff'],h['seed'],h['method']):h for h in heads};rankchecks=0
    for ref in read(OUT/'source_testing_features.json'):
        cutoff=ref['cutoff'];seed=ref['seed'];d=arrays(ref);src=arrays(ui[(cutoff,seed,METHODS[0])]);xx=independent_inputs(d['features'],d['market_features'],d['gate'],src)
        for method,x in zip(METHODS,xx+[xx[-1]]):
            key=(cutoff,seed,method);u=np.asarray(ui[key]['coefficients']);w=np.asarray(wi[key]['coefficients']);di=float(w[-1]-u[-1])
            for row,idx in zip(x,d['row_index']):
                zu=math.fsum(float(a)*float(b) for a,b in zip(row,u[:-1]))+float(u[-1]);zw=math.fsum(float(a)*float(b) for a,b in zip(row,w[:-1]))+float(w[-1]);ds=math.fsum(float(a)*(float(b)-float(c)) for a,b,c in zip(row,w[:-1],u[:-1]));zi=zu+di;zs=zw-di;t=ti.loc[(cutoff,method,seed,idx)]
                for attr,value in [('uniform_logit',zu),('weighted_logit',zw),('intercept_hybrid_logit',zi),('slopes_hybrid_logit',zs),('intercept_change',di),('slopes_change',ds)]:eq(getattr(t,attr),value)
                residual=zw-zu-di-ds;assert abs(residual)<1e-12 and abs(t.identity_residual)<1e-12;maxidentity=max(maxidentity,abs(residual),abs(t.identity_residual))
                for history,z in [(ANNUAL,zu),(WEIGHTED,zw),(INTERCEPT,zi),(SLOPES,zs)]:
                    a=mi.loc[(history,method,seed,idx)];p=float(expit(z));gap=abs(p-a.probability);assert gap<1e-10;maxgap=max(maxgap,gap);assert a.direction_up==int(p>.5) and a.score==a.probability
                    if history in NEW:
                        assert a.cutoff==cutoff==f'{int(a.date[:4])-1}-12-31'<a.date and a.joint_completed==obs.joint_completed.iloc[idx] and a.actual==obs.exec_return.iloc[idx] and a.actual_up==int(a.actual>0);count+=1
            # Constant intercept changes preserve ranking only within a fixed annual seed model.
            for a,b in [(INTERCEPT,ANNUAL),(SLOPES,WEIGHTED)]:
                pa=models[models.history.eq(a)&models.method.eq(method)&models.seed.eq(seed)&models.cutoff.eq(cutoff)].sort_values('row_index').probability.to_numpy();pb=models[models.history.eq(b)&models.method.eq(method)&models.seed.eq(seed)&models.cutoff.eq(cutoff)].sort_values('row_index').probability.to_numpy();np.testing.assert_array_equal(np.sign(pa[:,None]-pa),np.sign(pb[:,None]-pb));rankchecks+=1
    assert count==6528 and rankchecks==144
    for history in NEW:
        for method in ['native_mse','training_frequency']:
            a=models[models.history.eq(history)&models.method.eq(method)].drop(columns='history').reset_index(drop=True);b=base[base.history.eq(ANNUAL)&base.method.eq(method)].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True,check_dtype=False)
    for (history,method,idx),g in models.groupby(['history','method','row_index'],sort=False):
        n=1 if method=='training_frequency' else 3;assert len(g)==n;r=ei.loc[(history,method,idx)];eq(r.score,math.fsum(g.score)/n)
        if method=='native_mse':assert pd.isna(r.probability) and r.direction_up==int(r.score>0)
        else:eq(r.probability,math.fsum(g.probability)/n);assert r.direction_up==int(r.probability>.5)
        if history in NEW and method not in METHODS:
            a=ei.loc[(ANNUAL,method,idx)];assert r.score==a.score and r.direction_up==a.direction_up
            if method=='training_frequency':assert r.probability==a.probability
    prov=csv('prediction_sources');assert len(prov)==144
    for r in prov.itertuples():
        h=hi[(r.history,r.cutoff,r.seed,r.method)];t=next(t for t in read(OUT/'source_testing_features.json') if (t['cutoff'],t['seed'])==(r.cutoff,r.seed));assert r.head_job==h['job'] and r.training_source==h['frozen_pipeline_file'] and r.testing_source==t['cache_file'] and r.n==len(arrays(t)['row_index'])
    joined=ensemble.merge(csv('weekly_context')[['row_index','date','state']],on=['row_index','date'],validate='many_to_one');metriccount=0
    for name,source,keys in [('ensemble_metrics',ensemble,['history','method']),('seed_metrics',models,['history','method','seed']),('state_metrics',joined,['history','method','state'])]:
        groups={k:g for k,g in source.groupby(keys,sort=False)}
        for row in csv(name).to_dict('records'):
            w=next(w for w in periods() if w['name']==row['period']);g=groups.get(tuple(row[k] for k in keys),source.iloc[:0]);g=g[g.date.between(w['start'],w['end'])]
            for k,x in independent_metric(g).items():eq(x,row[k])
            metriccount+=1
    return dict(independent_learned_seed_predictions=count,maximum_probability_gap=maxgap,maximum_logit_identity_residual=maxidentity,annual_seed_rank_checks=rankchecks,independent_metric_cells=metriccount,old_histories_preserved=28,exact_native_frequency_controls=True)

def check_diagnostics():
    weekly=csv('weekly_policy_effects');e=csv('ensemble_predictions').set_index(['history','method','row_index']);assert len(weekly)==2176
    for r in weekly.itertuples():
        a=e.loc[(r.history,r.method,r.row_index)];assert a.probability==r.probability and a.actual_up==r.actual_up
        for label,history in [('annual',ANNUAL),('weighted',WEIGHTED),('quarter4',QUARTER)]:
            b=e.loc[(history,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;case='recovery' if good and not old else 'regression' if old and not good else 'stable_correct' if good else 'stable_wrong'
            assert getattr(r,label+'_probability')==b.probability and getattr(r,'case_vs_'+label)==case and getattr(r,'changed_vs_'+label)==(a.probability!=b.probability);eq(getattr(r,'brier_vs_'+label),(a.probability-a.actual_up)**2-(b.probability-b.actual_up)**2)
    for r in csv('reference_comparisons').itertuples():
        w=next(w for w in periods() if w['name']==r.period);g=weekly[weekly.history.eq(r.history)&weekly.method.eq(r.method)&weekly.date.between(w['start'],w['end'])];label='annual' if r.reference_history==ANNUAL else 'weighted' if r.reference_history==WEIGHTED else 'quarter4';assert r.n==len(g) and r.changed_probability_weeks==sum(g['changed_vs_'+label]) and r.recoveries==sum(g['case_vs_'+label]=='recovery') and r.regressions==sum(g['case_vs_'+label]=='regression');eq(r.brier_difference,math.fsum(g['brier_vs_'+label])/len(g))
    factor=csv('weekly_factorial_effects');summary=csv('factorial_effects');assert len(factor)==2176 and len(summary)==72
    for r in factor.itertuples():
        losses=[]
        for h,col in [(ANNUAL,'uniform_loss'),(INTERCEPT,'intercept_loss'),(SLOPES,'slopes_loss'),(WEIGHTED,'weighted_loss')]:
            a=e.loc[(h,r.method,r.row_index)];loss=float(a.direction_up!=a.actual_up) if r.metric=='direction_error' else (a.probability-a.actual_up)**2;eq(getattr(r,col),loss);losses.append(loss)
        u,i,s,w=losses;expected=[w-u,i-u,s-u,w-s,w-i,w-i-s+u,(i-u+w-s)/2,(s-u+w-i)/2]
        for col,value in zip(['total','intercept_at_uniform_slopes','slopes_at_uniform_intercept','intercept_at_weighted_slopes','slopes_at_weighted_intercept','interaction','symmetric_intercept','symmetric_slopes'],expected):eq(getattr(r,col),value)
        eq(r.symmetric_intercept+r.symmetric_slopes,r.total)
    for r in summary.itertuples():
        w=next(w for w in periods() if w['name']==r.period);g=factor[factor.method.eq(r.method)&factor.metric.eq(r.metric)&factor.date.between(w['start'],w['end'])];assert r.n==len(g)
        for col in ['total','intercept_at_uniform_slopes','slopes_at_uniform_intercept','intercept_at_weighted_slopes','slopes_at_weighted_intercept','interaction','symmetric_intercept','symmetric_slopes']:eq(getattr(r,col),math.fsum(g[col])/len(g))
    same_tables(csv('all_2026_cases'),weekly[weekly.year.eq(2026)])
    return dict(independent_weekly_effects=2176,independent_factorial_weekly_cells=2176,independent_factorial_period_cells=72)
'''+stats+'''
def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();started=now();last=read(OUT/'contract_verification.json')['completed_utc']
    for phase in ['assembly','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    source,gap=audit_sources();assembly=check_assembly();print('Independent144exactassemblies and144sourceendpointreplays PASS.',flush=True)
    pred=check_predictions();diag=check_diagnostics();check_statistics();assert old_evidence()==prep['old_evidence'];check_frozen()
    save(OUT/'verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=6143,new_head_fits=0,new_neural_fits=0,new_feature_inference=0,annual_source_jobs=18,endpoint_head_replays=144,maximum_frozen_design_gap=gap,independent_comparisons=72,**assembly,**pred,**diag));print('R48independentpredictions,metrics,logit/factorialdiagnostics and72comparisons PASS.',flush=True)
if __name__=='__main__':main()
'''
(P/'research_v48/verify48.py').write_text(code,encoding='utf-8');print('R48 verification source staged.')
