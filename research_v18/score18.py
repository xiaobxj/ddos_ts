from common18 import *

def main():
    check_frozen();fitting=check_phase('training');assert fitting['all_converged'] and fitting['new_primary_fits']==30
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scoring';run=manifest('scoring');save(path,run)
    base=pd.read_csv(OUT/'classification_baselines.csv',float_precision='round_trip');parents={h['job']:h for h in read(OUT/'parent_heads.json')}
    refs={r['job']:r for r in read(V16/'results/validation_features.json')};contexts={r['cutoff']:r for r in read(OUT/'market_contexts.json')}
    forecasts=[];parts=[];diagnostics=[]
    for h in read(OUT/'heads.json'):
        context=contexts[h['cutoff']];x=inputs(h,'validation',context,parents,refs);theta=np.asarray(h['coefficients']);z=design(x)@theta;p=probability(z)
        g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,context['validation_rows'])
        g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        width=h['representation_dimensions'];representation=x[:,:width]@theta[:width];market=x[:,width:]@theta[width:-1]
        np.testing.assert_allclose(z,representation+market+theta[-1],rtol=0,atol=1e-12)
        for row,rr,mm in zip(g.itertuples(),representation,market):parts.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],date=row.date,row_index=row.row_index,
            representation_logit=float(rr),market_logit=float(mm),intercept=float(theta[-1]),logit=float(row.logit)))
        for split,values in [('training',load_npz(h)['standardized']),('validation',x)]:
            rr=values[:,:width]@theta[:width];mm=values[:,width:]@theta[width:-1]
            diagnostics.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],split=split,n=len(values),
                representation_mean=float(rr.mean()),representation_std=float(rr.std()),market_mean=float(mm.mean()),market_std=float(mm.std()),
                component_correlation=float(np.corrcoef(rr,mm)[0,1]) if rr.std()>0 and mm.std()>0 else None,
                intercept=float(theta[-1]),total_logit_mean=float((rr+mm+theta[-1]).mean()),total_logit_std=float((rr+mm).std())))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==1305
    old=pd.read_csv(V17/'results/model_predictions.csv',float_precision='round_trip');models=pd.concat([old,new],ignore_index=True);assert len(models)==4176
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),joint_completed=('joint_completed','first'),
        actual_return=('actual_return','first'),actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index()
    e['direction_up']=e.probability>.5
    ensemble=pd.concat([pd.read_csv(V17/'results/ensemble_predictions.csv',float_precision='round_trip'),e],ignore_index=True)
    assert len(ensemble)==2610 and ensemble.groupby('method').size().eq(261).all()
    tables=dict(new_model_predictions=new,model_predictions=models,ensemble_predictions=ensemble,logit_components=pd.DataFrame(parts),component_summary=pd.DataFrame(diagnostics))
    files=[]
    for name,table in tables.items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    finish(run,files,new_probability_forecasts=1305,reused_model_records=2871,model_records=4176,ensemble_records=2610,heldout_dates=261,state_expert_switching=False)
    print('Scored1305 new probabilities on261unchanged dates, retained2871model records;10methods total.',flush=True)

if __name__=='__main__':main()
