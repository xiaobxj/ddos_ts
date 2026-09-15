from common29 import *
from contract29 import checks
import evaluate29 as evaluation
import verify15 as independent
from verify28 import independent_inputs
from scipy.special import expit

def main():
    legacy.initialize();started=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for key in ['source_sha256','protocol_sha256','input_sha256']:assert run[key]==prep[key]
    result,trigger_audit=checks();pd.testing.assert_frame_equal(trigger_audit,csv('independent_trigger_checks'),check_dtype=False)
    obs,price,targets=data();models=read(OUT/'source_models.json');heads=read(OUT/'source_heads.json');refs=read(OUT/'testing_features.json');pred=csv('model_predictions');ensemble=csv('ensemble_predictions');extra=csv('extension_predictions');missing=csv('extension_membership');scale=read(V28/'results/training_scales.json');replays=[];maxgap=0.;featuregap=0.
    for r in models:
        state=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==sha(V28/'protocol.json') and state['history']=='rolling5' and state['epoch']==20 and state['cutoff']==r['cutoff'] and state['seed']==r['seed']
        assert training.object_hash(state['state_dict'])==r['model_sha256'] and training.object_hash(state['optimizer_state_dict'])==r['optimizer_sha256']
        assert state['scales']==scale['rolling5_'+r['cutoff']]
    for t in refs:
        cutoff=t['cutoff'];seed=t['seed'];r=next(r for r in models if r['cutoff']==cutoff and r['seed']==seed);model,state=load_model(r);d=arrays(t);rows=missing[missing.cutoff.eq(cutoff)].sort_values('row_index').row_index.to_numpy(int);np.testing.assert_array_equal(rows,d['row_index'])
        assert obs.date.iloc[rows].gt(fold_for(cutoff)['end']).all() and obs.joint_completed.iloc[rows].le(cfg()['label_end']).all()
        f,native=forecasts(model,rows,scale[f'rolling5_{cutoff}']);np.testing.assert_array_equal(f,d['features']);np.testing.assert_array_equal(native,d['native'])
        hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED];training_data=arrays(hs[0]);xx=independent_inputs(f,market(price,obs,rows),gates(price,obs,rows),training_data)
        for x,key in zip(xx[:3],['x18','x19','x23']):
            g=float(np.max(np.abs(x-d[key])));featuregap=max(featuregap,g);assert g<1e-9
        for h,x in zip(hs,xx):
            p=expit(x@np.asarray(h['coefficients'])[:-1]+h['coefficients'][-1]);g=extra[extra.cutoff.eq(cutoff)&extra.seed.eq(seed)&extra.method.eq(h['method'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,rows);gap=float(np.max(np.abs(p-g.probability)));assert gap<1e-10;maxgap=max(maxgap,gap);np.testing.assert_array_equal(g.direction_up,(p>.5).astype(int))
        g=extra[extra.cutoff.eq(cutoff)&extra.seed.eq(seed)&extra.method.eq('native_mse')].sort_values('row_index');np.testing.assert_allclose(g.score,native,rtol=0,atol=1e-14);np.testing.assert_array_equal(g.direction_up,(native>0).astype(int));assert g.probability.isna().all()
        tr=training_rows(obs,cutoff);g=extra[extra.cutoff.eq(cutoff)&extra.method.eq('training_frequency')];np.testing.assert_allclose(g.probability,(targets['returns'][tr]>0).mean(),rtol=0,atol=1e-14)
        assert training.object_hash(model.state_dict())==r['model_sha256'];replays.append(dict(cutoff=cutoff,seed=seed,n=len(rows),training_unchanged=True,features_bitwise_equal=True,native_bitwise_equal=True));del model;torch.cuda.empty_cache()
    assert len(refs)==3*missing.cutoff.nunique() and len(extra)==16*len(missing)
    archive=archive_predictions();keys=['history','cutoff','row_index','method','seed'];assert not len(archive.merge(extra,on=keys));unique=pd.concat([archive,extra],ignore_index=True)
    # Independently select each source model by the frozen per-date route.
    route=csv('routing');candidate=pred[pred.history.eq('state90')].reset_index(drop=True);assert len(candidate)==4352
    chosen=candidate.drop(columns='history').merge(route[['row_index','date','cutoff']],on=['row_index','date','cutoff'],validate='many_to_one');assert len(chosen)==4352
    matched=candidate.drop(columns='history').merge(unique.drop(columns='history'),on=['cutoff','row_index','method','seed'],suffixes=('_actual','_source'),validate='one_to_one');assert len(matched)==4352
    for name in ['date','joint_completed','year','actual','actual_up','score','probability','direction_up']:
        a=matched[name+'_actual'];b=matched[name+'_source']
        if name in ['date','joint_completed']:assert a.equals(b)
        else:np.testing.assert_allclose(a,b,rtol=0,atol=1e-14,equal_nan=True)
    np.testing.assert_array_equal(candidate.actual_up,(targets['returns'][candidate.row_index]>0).astype(int));np.testing.assert_allclose(candidate.actual,targets['returns'][candidate.row_index],rtol=0,atol=1e-14)
    routed=route_predictions(unique);pd.testing.assert_frame_equal(candidate,routed,check_dtype=False,atol=1e-14,rtol=0)
    base,base_e=baselines();pd.testing.assert_frame_equal(pred[pred.history.ne('state90')].reset_index(drop=True),base,check_dtype=False,atol=1e-14,rtol=0);pd.testing.assert_frame_equal(ensemble[ensemble.history.ne('state90')].reset_index(drop=True),base_e,check_dtype=False,atol=1e-14,rtol=0)
    pd.testing.assert_frame_equal(ensemble_from(pred),ensemble,check_dtype=False,atol=1e-14,rtol=0)
    tables,pairs=evaluation.compute(pred,ensemble)
    for name,t in tables.items():pd.testing.assert_frame_equal(t,csv(name),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));assert len(pairs)==28
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['history','method','year']);independent.verify_metric_table(pred[pred.seed.ne(-1)],csv('seed_yearly_metrics'),['history','method','seed','year'])
    for w in evaluation.windows():
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['history','method']);independent.verify_metric_table(evaluation.select(pred[pred.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['history','method','seed'])
    assert len(pred)==17408 and len(ensemble)==6528;assert old_evidence()==prep['old_evidence'];check_frozen();pd.DataFrame(replays,columns=['cutoff','seed','n','training_unchanged','features_bitwise_equal','native_bitwise_equal']).to_csv(OUT/'extension_replays.csv',index=False)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=4522,independent_state=result,reused_checkpoint_states=len(models),extension_replays=len(replays),new_neural_fits=0,new_head_fits=0,maximum_feature_algebra_gap=featuregap,maximum_probability_gap=maxgap,independent_metrics=True,all28comparisons_recomputed=True,archived_predictions_preserved=True,artifacts={'extension_replays.csv':sha(OUT/'extension_replays.csv')}));print('Verification PASS: state decisions, model routing, extensions, metrics and old evidence.',flush=True)
if __name__=='__main__':main()
