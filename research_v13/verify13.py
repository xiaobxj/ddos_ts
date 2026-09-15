from common13 import *
import evaluate13 as evaluation

def compare_numbers(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare_numbers(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_numbers(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)

def main():
    legacy.initialize();start=time.time();prep=check_frozen()
    obs=pd.read_csv(OUT/'observation_table.csv');scales=read(OUT/'training_scales.json')
    rolling=pd.read_csv(OUT/'rolling_seed_predictions.csv');outer=pd.read_csv(OUT/'archived_predictions.csv')
    curves=pd.read_csv(OUT/'training_curves.csv');losses=pd.read_csv(OUT/'final_training_losses.csv')
    refs=read(OUT/'inner_models.json')+read(OUT/'archived_models.json');assert len(refs)==18
    max_prediction=0.;max_loss=0.;prediction_count=0;epoch_count=0
    for ref in refs:
        model,state=load_model(ref);cutoff=ref['cutoff'];seed=ref['seed']
        assert state['scales']==scales[cutoff]
        tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy())
        assert state['train_n']==len(tr)
        with np.load(V5/'cache/targets.npz') as raw:
            for key in ['returns','auxiliary']:
                target=raw[key][tr].astype(float)
                np.testing.assert_array_equal(target.mean(axis=0),np.asarray(scales[cutoff][key+'_mean']))
                np.testing.assert_array_equal(np.maximum(target.std(axis=0),1e-6),np.asarray(scales[cutoff][key+'_sd']))
        if cutoff in cfg()['new_training_cutoffs']:
            assert object_hash(state['optimizer_state_dict'])==ref['optimizer_sha256']
            assert object_hash({k:state[k] for k in ['numpy_rng_state','cpu_rng_state','cuda_rng_state']})==ref['rng_sha256']
            c=curves[curves.cutoff.eq(cutoff)&curves.seed.eq(seed)].sort_values('epoch');assert len(c)==20
            rng=np.random.default_rng(seed)
            for row in c.itertuples():
                order=rng.permutation(len(tr));batches=prior.batches(order)
                assert array_hash(order)==row.order_sha256
                assert object_hash([array_hash(b) for b in batches])==row.boundary_sha256
                assert row.presentations==len(tr) and row.steps==len(batches) and row.last_batch_n==len(batches[-1])
                assert row.lr==.001 and row.weight_decay==.1;epoch_count+=1
            assert object_hash(rng.bit_generator.state)==object_hash(state['numpy_rng_state'])
            for value in state['optimizer_state_dict']['state'].values():assert int(value['step'])==20*((len(tr)+127)//128)
            values,labels,_,_=load_training(cutoff);replayed=prior.training_loss(model,values,labels)
            old=losses[losses.cutoff.eq(cutoff)&losses.seed.eq(seed)].iloc[0]
            for k,v in replayed.items():
                error=abs(v-old[k]);max_loss=max(max_loss,error);assert error<1e-12
            del values,labels
        for pool,key in [(rolling,'inner_cutoff'),(outer,'cutoff')]:
            g=pool[pool[key].eq(cutoff)&pool.seed.eq(seed)].sort_values('row_index')
            if not len(g):continue
            estimates=forecast(model,g.row_index.to_numpy(),scales[cutoff])
            error=float(np.max(np.abs(estimates-g.predicted_return.to_numpy())))
            max_prediction=max(max_prediction,error);assert error<1e-10
            np.testing.assert_allclose(g.actual,obs.exec_return.iloc[g.row_index],rtol=0,atol=1e-12)
            assert g.date.gt(cutoff).all();prediction_count+=len(g)
        del model;torch.cuda.empty_cache()
        print(f'Neural replay {cutoff} seed {seed}: PASS',flush=True)
    assert prediction_count==1068 and epoch_count==180
    # Independent maturity membership and sufficient-statistic reconstruction.
    ensemble=rolling.groupby('date').agg(inner_cutoff=('inner_cutoff','first'),row_index=('row_index','first'),
        predicted_return=('predicted_return','mean'),training_mean=('training_mean','first'),actual=('actual','first'),
        joint_completed=('joint_completed','first')).reset_index()
    coefficient_rows=read(OUT/'coefficients.json');jack=pd.read_csv(OUT/'calibration_leave_year_out.csv')
    contributions=pd.read_csv(OUT/'calibration_year_contributions.csv');max_alpha=0.;membership_count=0
    for r in coefficient_rows:
        cutoff=r['outer_cutoff'];g=ensemble[ensemble.date.ge(f'{int(cutoff[:4])-2}-01-01')&ensemble.date.le(cutoff)&ensemble.joint_completed.le(cutoff)]
        expected=np.flatnonzero((obs.date.ge(f'{int(cutoff[:4])-2}-01-01')&obs.date.le(cutoff)&obs.weekday.eq(4)&obs.joint_completed.le(cutoff)).to_numpy())
        np.testing.assert_array_equal(g.row_index,expected);assert len(g)==r['n']
        saved=pd.read_csv(ROOT/r['input_file']);assert sha(ROOT/r['input_file'])==r['input_sha256']
        np.testing.assert_array_equal(saved.row_index,expected)
        for k in ['predicted_return','training_mean','actual']:np.testing.assert_allclose(saved[k],g[k],rtol=0,atol=1e-14)
        x=g.predicted_return.to_numpy()-g.training_mean.to_numpy();z=g.actual.to_numpy()-g.training_mean.to_numpy()
        numerator=float(np.sum(x*z));denominator=float(np.sum(x*x));alpha=float(max(0.,min(1.,numerator/denominator)))
        assert len(g)>=52 and g.date.str[:4].nunique()==3 and denominator>1e-18 and not r['fallback']
        for k,value in [('numerator',numerator),('denominator',denominator),('raw_alpha',numerator/denominator),('alpha',alpha)]:assert abs(r[k]-value)<1e-12
        assert r['latest_label_completion']==g.joint_completed.max()
        max_alpha=max(max_alpha,abs(r['alpha']-alpha));membership_count+=len(g)
        for year,h in g.groupby(g.date.str[:4]):
            v=contributions[contributions.outer_cutoff.eq(cutoff)&contributions.year.eq(int(year))].iloc[0]
            xx=h.predicted_return.to_numpy()-h.training_mean.to_numpy();zz=h.actual.to_numpy()-h.training_mean.to_numpy()
            assert v.n==len(h) and abs(v.numerator-float(xx@zz))<1e-12 and abs(v.denominator-float(xx@xx))<1e-12
            keep=g[g.date.str[:4].ne(year)];xx=keep.predicted_return.to_numpy()-keep.training_mean.to_numpy();zz=keep.actual.to_numpy()-keep.training_mean.to_numpy()
            v=jack[jack.outer_cutoff.eq(cutoff)&jack.omitted_year.eq(int(year))].iloc[0]
            raw_alpha=float((xx@zz)/(xx@xx));fallback=len(keep)<52
            assert v.n==len(keep) and bool(v.fallback)==fallback and v.diagnostic_only
            assert abs(v.raw_alpha-raw_alpha)<1e-12 and abs(v.alpha-(0. if fallback else np.clip(raw_alpha,0,1)))<1e-12
    assert membership_count==407
    # Coefficients were frozen before transforming/evaluating any outer forecasts.
    phases=['preparation','contract_verification','training','rolling','calibration','scoring','evaluation']
    times=[]
    for phase in phases:
        r=read(OUT/(phase+'.json' if phase=='contract_verification' else phase+'_manifest.json'))
        times.append(r.get('finished_utc',r.get('completed_utc')))
        if phase not in ['preparation','contract_verification']:
            assert r['started_utc']>=times[-2]
            for n,d in r.get('artifacts',{}).items():assert sha(ROOT/n if n.startswith('results') else OUT/n)==d,n
    assert times==sorted(times)
    predicted=pd.read_csv(OUT/'outer_seed_predictions.csv');averaged=pd.read_csv(OUT/'outer_ensemble_predictions.csv')
    alphas={r['outer_cutoff']:r['alpha'] for r in coefficient_rows};max_transform=0.
    for method,g in predicted.groupby('method'):
        merged=g.merge(outer,on=['cutoff','seed','row_index','date'],suffixes=('_new','_old'),validate='one_to_one');assert len(merged)==423
        a=merged.cutoff.map(alphas).to_numpy() if method=='rolling_shrink' else (1. if method=='archived20' else .5)
        m=merged.training_mean_old.to_numpy();p=merged.predicted_return_old.to_numpy();expected=m+a*(p-m)
        error=float(np.max(np.abs(expected-merged.predicted_return_new.to_numpy())));max_transform=max(max_transform,error);assert error<1e-14
        np.testing.assert_allclose(merged.alpha,a,rtol=0,atol=1e-14)
        agg=g.groupby('date').predicted_return.mean();e=averaged[averaged.method.eq(method)].sort_values('date')
        np.testing.assert_allclose(agg,e.predicted_return,rtol=0,atol=1e-14)
        np.testing.assert_allclose(e.predicted_return,e.training_mean+e.alpha*(e.original_prediction-e.training_mean),rtol=0,atol=1e-14)
    table,years,seeds,pairs,assessment=evaluation.compute()
    for actual,name in [(table,'ensemble_metrics.csv'),(years,'yearly_metrics.csv'),(seeds,'seed_metrics.csv')]:
        expected=pd.read_csv(OUT/name);pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
    compare_numbers(pairs,read(OUT/'primary_comparisons.json'));compare_numbers(assessment,read(OUT/'assessment.json'))
    assert old_evidence()==prep['old_evidence']
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,model_states_replayed=18,
        training_loss_passes=9,epoch_orders_and_batches=180,rolling_seed_forecasts=645,outer_base_seed_forecasts=423,
        outer_transformed_seed_forecasts=1269,calibration_memberships=407,coefficients_recomputed=3,
        leave_year_coefficients_recomputed=9,primary_comparisons_recomputed=4,previous_files_preserved=2341,
        max_prediction_error=max_prediction,max_training_loss_error=max_loss,max_alpha_error=max_alpha,max_transform_error=max_transform,
        source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'protocol.json'))
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
