from common21 import *

def main():
    check_frozen();fit=check_phase('training');assert fit['all_converged'] and fit['new_hmm_fits']==6 and fit['new_primary_fits']==30
    path=OUT/'scoring_manifest.json';assert not path.exists();run=manifest('scoring');save(path,run)
    obs,price,returns=data();daily=daily_returns(price);contexts={c['cutoff']:c for c in read(OUT/'hmm_contexts.json')};gates={};files=[];daily_rows=[];signals=[];valrefs=[]
    for f in cfg()['folds']:
        c=contexts[f['cutoff']];d=load_npz(c);a,b=daily_extent(price,f);initial=d['filtered'][-1]@d['transition']
        z=forward(daily[a:b],d['variances'],d['transition'],initial=initial);tr,te=indices(obs,f);positions=obs.anchor.iloc[te].to_numpy(int)-a-1;p=z['filtered'][positions,1];gates[f['cutoff']]=2*p-1
        path=CACHE/f"validation_hmm_{f['cutoff']}.npz";np.savez_compressed(path,**z,initial=initial,anchor=np.arange(a+1,b+1),daily_return=daily[a:b],row_index=te,high_probability=p);files.append(path)
        valrefs.append(dict(cutoff=f['cutoff'],**reference(path)))
        for j,anchor in enumerate(range(a+1,b+1)):daily_rows.append(dict(cutoff=f['cutoff'],anchor=anchor,date=price.date.iloc[anchor],daily_return=daily[anchor-1],high_probability=float(z['filtered'][j,1]),prior_high_probability=float(z['predicted'][j,1])))
        for i,row in enumerate(te):signals.append(dict(cutoff=f['cutoff'],row_index=int(row),date=obs.date.iloc[row],high_probability=float(p[i]),gate=float(2*p[i]-1),hmm_bucket=buckets(p)[i]))
    base=pd.read_csv(OUT/'classification_baselines.csv',float_precision='round_trip');sources={s['job']:s for s in read(OUT/'source_heads.json')};forecasts=[];components=[];summaries=[]
    for h in read(OUT/'heads.json'):
        d=load_npz(h);u=gates[h['cutoff']];theta=np.asarray(h['coefficients'])
        if h['source_job']:
            x,rows=base_inputs(sources[h['source_job']],'validation');xx,extra=apply_interaction(x,u,d)
        else:
            rows=base[base.cutoff.eq(h['cutoff'])].row_index.to_numpy(int);xx=((u-float(d['gate_mean']))/float(d['gate_sd']))[:,None]
        z=design(xx)@theta;p=probability(z);g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,rows)
        g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        if h['source_job']:
            for i,row in enumerate(rows):components.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],row_index=int(row),date=obs.date.iloc[row],gate=float(u[i]),
                parent_representation_signal=float(extra['signal'][i]),product=float(extra['product'][i]),interaction_feature=float(xx[i,-1]),
                refitted_additive_logit=float(x[i]@theta[:-2]+theta[-1]),interaction_logit=float(xx[i,-1]*theta[-2]),logit=float(z[i])))
            for split,values in [('training',d['standardized']),('validation',xx)]:
                aa=values[:,:-1]@theta[:-2]+theta[-1];bb=values[:,-1]*theta[-2]
                summaries.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],split=split,n=len(values),interaction_coefficient=float(theta[-2]),raw_product_coefficient=float(theta[-2]/d['residual_sd']),
                    interaction_feature_mean=float(values[:,-1].mean()),interaction_feature_std=float(values[:,-1].std()),refitted_additive_mean=float(aa.mean()),interaction_logit_mean=float(bb.mean()),interaction_logit_std=float(bb.std()),total_logit_mean=float((aa+bb).mean())))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==1305
    models=pd.concat([pd.read_csv(V20/'results/model_predictions.csv',float_precision='round_trip'),new],ignore_index=True)
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),joint_completed=('joint_completed','first'),actual_return=('actual_return','first'),
        actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index();e['direction_up']=e.probability>.5
    ensemble=pd.concat([pd.read_csv(V20/'results/ensemble_predictions.csv',float_precision='round_trip'),e],ignore_index=True)
    assert len(models)==8613 and len(ensemble)==5220 and ensemble.groupby('method').size().eq(261).all() and len(daily_rows)==1462
    for name,table in dict(new_model_predictions=new,model_predictions=models,ensemble_predictions=ensemble,hmm_daily_states=pd.DataFrame(daily_rows),hmm_signal_states=pd.DataFrame(signals),
        interaction_components=pd.DataFrame(components),component_summary=pd.DataFrame(summaries)).items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    path=OUT/'hmm_validation_contexts.json';save(path,valrefs);files.append(path)
    finish(run,files,new_probability_forecasts=1305,reused_model_records=7308,model_records=8613,ensemble_records=5220,heldout_dates=261,heldout_daily_steps=1462,interaction_components=1044)
    print('Scored1305new model probabilities;20methods on261unchanged dates;1462frozen-parameter daily filter steps.',flush=True)

if __name__=='__main__':main()
