from common19 import *

def main():
    check_frozen();fitting=check_phase('training');assert fitting['all_converged'] and fitting['new_primary_fits']==24
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scoring';run=manifest('scoring');save(path,run)
    base=pd.read_csv(OUT/'classification_baselines.csv',float_precision='round_trip');sources={h['job']:h for h in read(OUT/'source_heads.json')}
    forecasts=[];components=[];summaries=[]
    for h in read(OUT/'heads.json'):
        d=load_npz(h);source=sources[h['source_job']];x,v,rows=inherited_inputs(source,'validation');xx,extra=apply_interaction(x,v,d)
        theta=np.asarray(h['coefficients']);z=design(xx)@theta;p=probability(z);additive=x@theta[:h['base_dimensions']]+theta[-1];interaction=xx[:,-1]*theta[-2]
        np.testing.assert_allclose(z,additive+interaction,rtol=0,atol=1e-12)
        g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,rows)
        g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        for i,row in enumerate(g.itertuples()):components.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],date=row.date,row_index=int(row.row_index),
            standardized_volatility=float(v[i]),parent_representation_signal=float(extra['signal'][i]),product=float(extra['product'][i]),standardized_interaction=float(extra['interaction'][i]),
            refitted_additive_logit=float(additive[i]),interaction_logit=float(interaction[i]),logit=float(z[i])))
        for split,values,vol,signal,product in [('training',d['standardized'],d['volatility'],d['signal'],d['product']),('validation',xx,v,extra['signal'],extra['product'])]:
            a=values[:,:h['base_dimensions']]@theta[:h['base_dimensions']]+theta[-1];b=values[:,-1]*theta[-2]
            summaries.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],split=split,n=len(values),
                volatility_mean=float(vol.mean()),signal_mean=float(signal.mean()),signal_std=float(signal.std()),product_mean=float(product.mean()),product_std=float(product.std()),
                interaction_feature_mean=float(values[:,-1].mean()),interaction_feature_std=float(values[:,-1].std()),
                refitted_additive_mean=float(a.mean()),refitted_additive_std=float(a.std()),interaction_logit_mean=float(b.mean()),interaction_logit_std=float(b.std()),
                total_logit_mean=float((a+b).mean()),total_logit_std=float((a+b).std())))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==1044
    models=pd.concat([pd.read_csv(V18/'results/model_predictions.csv',float_precision='round_trip'),new],ignore_index=True);assert len(models)==5220
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),joint_completed=('joint_completed','first'),
        actual_return=('actual_return','first'),actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index()
    e['direction_up']=e.probability>.5;ensemble=pd.concat([pd.read_csv(V18/'results/ensemble_predictions.csv',float_precision='round_trip'),e],ignore_index=True)
    assert len(ensemble)==3132 and ensemble.groupby('method').size().eq(261).all()
    tables=dict(new_model_predictions=new,model_predictions=models,ensemble_predictions=ensemble,interaction_components=pd.DataFrame(components),component_summary=pd.DataFrame(summaries));files=[]
    for name,table in tables.items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    finish(run,files,new_probability_forecasts=1044,reused_model_records=4176,model_records=5220,ensemble_records=3132,heldout_dates=261,interaction_components=1044)
    print('Scored1044new probabilities;12methods on261unchanged dates. No held-out transform fitting.',flush=True)

if __name__=='__main__':main()
