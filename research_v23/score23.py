from common23 import *

def main():
    check_frozen();fit=check_phase('training');assert fit['all_converged'] and fit['new_primary_fits']==30
    path=OUT/'scoring_manifest.json';assert not path.exists();run=manifest('scoring');save(path,run);obs,price,returns=data();files=[];signals=[];daily=[];stability=[];runs=[]
    for f in cfg()['folds']:
        tr,te=indices(obs,f);a,b=daily_extent(price,f);state=features(price.iloc[:b+1]);sg=signal_rows(state,obs,te,f['cutoff']);signals.append(sg)
        ds=state.iloc[a+1:b+1].copy();ds.insert(0,'cutoff',f['cutoff']);daily.append(ds)
        for gate in GATES:
            d,rs=run_diagnostics(ds,gate);stability.append(dict(cutoff=f['cutoff'],split='validation',gate=gate,**d));runs.extend(dict(cutoff=f['cutoff'],split='validation',gate=gate,**r) for r in rs)
    for name,table in [('validation_signal_states',pd.concat(signals,ignore_index=True)),('validation_daily_states',pd.concat(daily,ignore_index=True)),('validation_stability',pd.DataFrame(stability)),('validation_state_runs',pd.DataFrame(runs))]:
        path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    base=pd.read_csv(OUT/'classification_baselines.csv',float_precision='round_trip');sources={s['job']:s for s in read(OUT/'source_heads.json')};forecasts=[];components=[];summaries=[];correlations=[]
    market_refs={c['cutoff']:c for c in read(V18/'results/market_contexts.json')}
    for h in read(OUT/'heads.json'):
        u,rows=gate_inputs(h,'validation');d=load_npz(h);theta=np.asarray(h['coefficients'])
        if h['source_job']:
            x,idx=base_inputs(sources[h['source_job']],'validation');np.testing.assert_array_equal(idx,rows);xx,extra=apply_interaction(x,u,d)
        else:xx=((u-float(d['gate_mean']))/float(d['gate_sd']))[:,None]
        z=design(xx)@theta;p=probability(z);g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,rows)
        g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        if h['source_job']:
            for i,row in enumerate(rows):components.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],row_index=int(row),date=obs.date.iloc[row],gate=float(u[i]),parent_representation_signal=float(extra['signal'][i]),product=float(extra['product'][i]),
                interaction_feature=float(xx[i,-1]),refitted_parent_logit=float(x[i]@theta[:-2]+theta[-1]),refitted_base_logit=float(x[i,:-1]@theta[:-3]+theta[-1]),refitted_original_interaction_logit=float(x[i,-1]*theta[-3]),interaction_logit=float(xx[i,-1]*theta[-2]),logit=float(z[i])))
            for split,values in [('training',d['standardized']),('validation',xx)]:
                aa=values[:,:-1]@theta[:-2]+theta[-1];bb=values[:,-1]*theta[-2]
                summaries.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],split=split,n=len(values),interaction_coefficient=float(theta[-2]),raw_product_coefficient=float(theta[-2]/d['residual_sd']),interaction_feature_mean=float(values[:,-1].mean()),
                    interaction_feature_std=float(values[:,-1].std()),refitted_parent_mean=float(aa.mean()),interaction_logit_mean=float(bb.mean()),interaction_logit_std=float(bb.std()),total_logit_mean=float((aa+bb).mean())))
    # State correlations are diagnostics, not additional fit/selection criteria.
    for f in cfg()['folds']:
        for split in ['training','validation']:
            old=pd.read_csv(V18/'results'/f'{split}_states.csv',float_precision='round_trip');old=old[old.cutoff.eq(f['cutoff'])].sort_values('row_index')
            for gate in GATES:
                u,rows=gate_inputs(dict(cutoff=f['cutoff'],gate=gate),split);np.testing.assert_array_equal(old.row_index,rows)
                state=pd.read_csv(OUT/f'{split}_signal_states.csv',float_precision='round_trip');state=state[state.cutoff.eq(f['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(state.row_index,rows)
                correlations.append(dict(cutoff=f['cutoff'],split=split,gate=gate,n=len(u),correlation_volatility20=float(np.corrcoef(u,old.volatility20)[0,1]),correlation_abs_trend60=float(np.corrcoef(u,abs(old.trend60))[0,1]),correlation_efficiency60=float(np.corrcoef(u,state.efficiency60)[0,1]),correlation_relativevol=float(np.corrcoef(u,state.relative_volatility)[0,1])))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==1305;models=pd.concat([pd.read_csv(V22/'results/model_predictions.csv',float_precision='round_trip'),new],ignore_index=True)
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),joint_completed=('joint_completed','first'),actual_return=('actual_return','first'),actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index();e['direction_up']=e.probability>.5
    ensemble=pd.concat([pd.read_csv(V22/'results/ensemble_predictions.csv',float_precision='round_trip'),e],ignore_index=True);assert len(models)==12528 and len(ensemble)==7569 and ensemble.groupby('method').size().eq(261).all()
    for name,table in dict(new_model_predictions=new,model_predictions=models,ensemble_predictions=ensemble,interaction_components=pd.DataFrame(components),component_summary=pd.DataFrame(summaries),state_correlations=pd.DataFrame(correlations)).items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    finish(run,files,new_probability_forecasts=1305,reused_model_records=11223,model_records=12528,ensemble_records=7569,heldout_dates=261,heldout_daily_steps=1462,interaction_components=1044)
    print('Scored1305new probabilities;29methods on261dates.',flush=True)

if __name__=='__main__':main()
