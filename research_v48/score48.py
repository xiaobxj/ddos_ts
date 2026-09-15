from common48 import *
def main():
    check_frozen();check_phase('assembly');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs=csv('observation_table');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');heads=read(OUT/'heads.json');uniform=read(OUT/'uniform_heads.json');weighted=read(OUT/'weighted_heads.json');parts=[];provenance=[];terms=[]
    ui={(h['cutoff'],h['seed'],h['method']):h for h in uniform};wi={(h['cutoff'],h['seed'],h['method']):h for h in weighted};hi={(h['history'],h['cutoff'],h['seed'],h['method']):h for h in heads}
    for ref in read(OUT/'source_testing_features.json'):
        d=arrays(ref);cutoff=ref['cutoff'];seed=ref['seed'];ids=d['row_index'];assert all(f'{int(date[:4])-1}-12-31'==cutoff for date in obs.date.iloc[ids])
        for method,key in zip(METHODS,['x18','x19','x23','x23']):
            k=(cutoff,seed,method);u=np.asarray(ui[k]['coefficients']);w=np.asarray(wi[k]['coefficients']);x=d[key];z0=design(x)@u;z1=design(x)@w;di=float(w[-1]-u[-1]);ds=x@(w[:-1]-u[:-1]);zz={}
            for history in NEW:
                h=hi[(history,*k)];z=design(x)@np.asarray(h['coefficients']);zz[history]=z;p=probability(z);parts.append(prediction_rows(obs,ids,history,cutoff,method,seed,p));provenance.append(dict(history=history,cutoff=cutoff,seed=seed,method=method,n=len(ids),training_source=h['frozen_pipeline_file'],testing_source=ref['cache_file'],head_job=h['job']))
            for j,idx in enumerate(ids):terms.append(dict(cutoff=cutoff,seed=seed,method=method,row_index=int(idx),date=obs.date.iloc[idx],uniform_logit=float(z0[j]),weighted_logit=float(z1[j]),intercept_hybrid_logit=float(zz[INTERCEPT][j]),slopes_hybrid_logit=float(zz[SLOPES][j]),intercept_change=di,slopes_change=float(ds[j]),identity_residual=float(z1[j]-z0[j]-di-ds[j])))
    for history in NEW:
        g=base[base.history.eq(ANNUAL)&~base.method.isin(METHODS)].copy();g['history']=history;parts.append(g)
    new=pd.concat(parts,ignore_index=True)[base.columns];assert len(new)==8704;models=pd.concat([base,new],ignore_index=True);assert len(models)==130560
    en=[]
    for (history,method,date),g in new.groupby(['history','method','date'],sort=False):
        assert len(g)==(1 if method=='training_frequency' else 3);r=g.iloc[0].to_dict();r.pop('seed');r['score']=float(g.score.mean());r['probability']=float(g.probability.mean()) if method!='native_mse' else np.nan;r['direction_up']=int(r['score']>0 if method=='native_mse' else r['probability']>.5);en.append(r)
    ensemble=pd.concat([be,pd.DataFrame(en)[be.columns]],ignore_index=True);assert len(ensemble)==48960 and ensemble.history.nunique()==30;files=[]
    for name,t in [('model_predictions',models),('ensemble_predictions',ensemble),('prediction_sources',pd.DataFrame(provenance)),('seed_logit_decomposition',pd.DataFrame(terms))]:p=OUT/f'{name}.csv';t.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_learned_seed_rows=6528,new_model_rows=8704,old_histories_preserved=28,new_head_changes_during_scoring=0,logit_cells=3264);print('Two fixed hybrids scored on272weeks;28oldhistories preserved.',flush=True)
if __name__=='__main__':main()
