from common24 import *

def main():
    check_frozen();fit=check_phase('training');assert fit['all_converged'] and fit['new_primary_fits']==48
    assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');save(OUT/'scoring_manifest.json',run);obs,price,returns=data()
    sources={s['job']:s for s in read(OUT/'source_heads.json')};parents={s['job']:s for s in read(V19/'results/heads.json')};base=csv('classification_baselines');forecasts=[];components=[];summaries=[]
    for h in read(OUT/'heads.json'):
        source=sources[h['source_job']];x,rows=inputs(source,'validation');xx=scaled_inputs(x,h['multiplier']);theta=np.asarray(h['coefficients']);beta=np.asarray(h['original_coefficients']);z=design(xx)@theta;p=probability(z);np.testing.assert_allclose(z,design(x)@beta,rtol=0,atol=1e-12)
        g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,rows);g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        parent_beta=np.asarray(parents[source['source_job']]['coefficients']);weak_beta=np.asarray(source['coefficients'])
        for i,row in enumerate(rows):components.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],row_index=int(row),date=obs.date.iloc[row],order_feature=float(x[i,-1]),refitted_base_logit=float(x[i,:-2]@beta[:-3]+beta[-1]),original_interaction_logit=float(x[i,-2]*beta[-3]),order_interaction_logit=float(x[i,-1]*beta[-2]),logit=float(z[i])))
        for split,values in [('training',load_npz(source)['standardized']),('validation',x)]:
            zz=design(values)@beta;oldz=design(values[:,:-1])@parent_beta;weakz=design(values)@weak_beta
            summaries.append(dict(job=h['job'],method=h['method'],source_method=source['method'],cutoff=h['cutoff'],seed=h['seed'],multiplier=h['multiplier'],split=split,n=len(values),order_coefficient=float(beta[-2]),original_interaction_coefficient=float(beta[-3]),order_logit_sd=float((values[:,-1]*beta[-2]).std()),parent_coefficient_l2_drift=float(np.linalg.norm(np.r_[beta[:-2],beta[-1]]-parent_beta)),rms_probability_change_vs_parent=float(np.sqrt(np.square(probability(zz)-probability(oldz)).mean())),rms_probability_change_vs_weak=float(np.sqrt(np.square(probability(zz)-probability(weakz)).mean()))))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==2088;models=pd.concat([pd.read_csv(V23/'results/model_predictions.csv',float_precision='round_trip'),new],ignore_index=True)
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),joint_completed=('joint_completed','first'),actual_return=('actual_return','first'),actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index();e['direction_up']=e.probability>.5
    ensemble=pd.concat([pd.read_csv(V23/'results/ensemble_predictions.csv',float_precision='round_trip'),e],ignore_index=True);assert len(models)==14616 and len(ensemble)==8613 and ensemble.groupby('method').size().eq(261).all()
    files=[]
    for name,frame in dict(new_model_predictions=new,model_predictions=models,ensemble_predictions=ensemble,interaction_components=pd.DataFrame(components),component_summary=pd.DataFrame(summaries)).items():path=OUT/f'{name}.csv';frame.to_csv(path,index=False);files.append(path)
    finish(run,files,new_probability_forecasts=2088,reused_model_records=12528,model_records=14616,ensemble_records=8613,interaction_components=2088)
    print('Scored2088new probabilities;33methods on261dates.',flush=True)
if __name__=='__main__':main()
