from common20 import *

def main():
    check_frozen();fitting=check_phase('training');assert fitting['all_converged'] and fitting['new_primary_fits']==48
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scoring';run=manifest('scoring');save(path,run)
    base=pd.read_csv(OUT/'classification_baselines.csv',float_precision='round_trip');sources={s['job']:s for s in read(OUT/'source_heads.json')}
    forecasts=[];components=[];distributions=[]
    for h in read(OUT/'heads.json'):
        source=sources[h['source_job']];xx,rows=source_inputs(source,h['source_round'],'validation');theta=np.asarray(h['coefficients']);z=design(xx)@theta;p=probability(z)
        g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,rows)
        g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        if h['interaction']:
            additive=xx[:,:-1]@theta[:-2]+theta[-1];interaction=xx[:,-1]*theta[-2]
            np.testing.assert_allclose(z,additive+interaction,rtol=0,atol=1e-12)
            for i,row in enumerate(g.itertuples()):components.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],date=row.date,row_index=int(row.row_index),
                inherited_interaction_feature=float(xx[i,-1]),refitted_additive_logit=float(additive[i]),interaction_logit=float(interaction[i]),logit=float(z[i])))
            full=load_npz(source)['standardized'];recent=load_npz(h)['standardized']
            for split,values in [('full_training',full),('recent_training',recent),('validation',xx)]:
                hh=values[:,-1];a=values[:,:-1]@theta[:-2]+theta[-1];b=hh*theta[-2]
                distributions.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],split=split,n=len(values),
                    interaction_feature_mean=float(hh.mean()),interaction_feature_std=float(hh.std()),interaction_feature_min=float(hh.min()),interaction_feature_max=float(hh.max()),
                    interaction_coefficient=float(theta[-2]),old_interaction_coefficient=float(source['coefficients'][-2]),refitted_additive_mean=float(a.mean()),
                    interaction_logit_mean=float(b.mean()),interaction_logit_std=float(b.std()),total_logit_mean=float((a+b).mean()),total_logit_std=float((a+b).std())))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==2088
    models=pd.concat([pd.read_csv(V19/'results/model_predictions.csv',float_precision='round_trip'),new],ignore_index=True);assert len(models)==7308
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),joint_completed=('joint_completed','first'),
        actual_return=('actual_return','first'),actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index()
    e['direction_up']=e.probability>.5;old=pd.read_csv(V19/'results/ensemble_predictions.csv',float_precision='round_trip')
    rates=pd.read_csv(OUT/'fold_summaries.csv',float_precision='round_trip').set_index('cutoff').recent_frequency
    freq=old[old.method.eq('training_frequency')].copy();freq['method']='recent_frequency';freq['probability']=freq.cutoff.map(rates);freq['score']=freq.probability;freq['direction_up']=freq.probability>.5
    ensemble=pd.concat([old,e,freq],ignore_index=True);assert len(ensemble)==4437 and ensemble.groupby('method').size().eq(261).all()
    tables=dict(new_model_predictions=new,model_predictions=models,ensemble_predictions=ensemble,recent_frequency_predictions=freq,
        interaction_components=pd.DataFrame(components),interaction_input_distribution=pd.DataFrame(distributions));files=[]
    for name,table in tables.items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    finish(run,files,new_probability_forecasts=2088,new_deterministic_baseline_forecasts=261,reused_model_records=5220,model_records=7308,ensemble_records=4437,heldout_dates=261,
        interaction_components=1044,interaction_distribution_groups=72)
    print('Scored2088new model probabilities plus261recent-frequency baseline values;17methods on unchanged261dates.',flush=True)

if __name__=='__main__':main()
