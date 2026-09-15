from common30 import *
from contract30 import coverage_checks,synthetic_cases,audit_policy
import evaluate30 as evaluation
import verify15 as independent
from verify28 import independent_inputs
from scipy.special import expit
import math

def independent_feedback(unique,obs,targets):
    rows=[]
    for (cutoff,i),g in unique.groupby(['cutoff','row_index'],sort=True):
        r=g[g.method.eq('learned_vol_interaction')].sort_values('seed');assert r.seed.tolist()==cfg()['seeds'];p=math.fsum(r.probability.tolist())/3;freq=g[g.method.eq('training_frequency')];assert len(freq)==1;tr=training_rows(obs,cutoff);q=int((targets['returns'][tr]>0).sum())/len(tr);assert abs(q-freq.probability.iloc[0])<1e-14
        assert cutoff<obs.date.iloc[i];rows.append(dict(row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],cutoff=cutoff,prediction_probability=p,frequency_probability=q,actual_up=int(targets['returns'][i]>0)))
    return pd.DataFrame(rows,columns=FEEDBACK_COLUMNS)

def main():
    legacy.initialize();started=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['extension','scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for key in ['source_sha256','protocol_sha256','input_sha256']:assert run[key]==prep[key]
    coverage_result=coverage_checks();assert len(synthetic_cases())==10
    policy=read(OUT/'policy_manifest.json');assert check_phase('extension')['finished_utc']<=policy['completed_utc']<=check_phase('scoring')['finished_utc']
    for n,d in policy['artifacts'].items():assert sha(ROOT/n)==d
    obs,price,targets=data();models=read(OUT/'source_models.json');heads=read(OUT/'source_heads.json');refs=read(OUT/'testing_features.json');pred=csv('model_predictions');ensemble=csv('ensemble_predictions');extra=csv('extension_predictions');missing=csv('extension_membership');scale=read(V28/'results/training_scales.json');replays=[];maxgap=0.;featuregap=0.
    for r in models:
        state=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True);assert state['protocol_sha256']==sha(V28/'protocol.json') and state['history']=='rolling5' and state['epoch']==20 and state['cutoff']==r['cutoff'] and state['seed']==r['seed']
        assert training.object_hash(state['state_dict'])==r['model_sha256'] and training.object_hash(state['optimizer_state_dict'])==r['optimizer_sha256'];assert state['scales']==scale['rolling5_'+r['cutoff']]
    for t in refs:
        cutoff=t['cutoff'];seed=t['seed'];r=next(r for r in models if r['cutoff']==cutoff and r['seed']==seed);model,_=load_model(r);d=arrays(t);rows=missing[missing.cutoff.eq(cutoff)].sort_values('row_index').row_index.to_numpy(int);np.testing.assert_array_equal(rows,d['row_index'])
        assert obs.date.iloc[rows].gt(fold_for(cutoff)['end']).all() and obs.joint_completed.iloc[rows].le(cfg()['label_end']).all()
        f,native=forecasts(model,rows,scale[f'rolling5_{cutoff}']);np.testing.assert_array_equal(f,d['features']);np.testing.assert_array_equal(native,d['native'])
        hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED];xx=independent_inputs(f,market(price,obs,rows),gates(price,obs,rows),arrays(hs[0]))
        for x,key in zip(xx[:3],['x18','x19','x23']):
            gap=float(np.max(np.abs(x-d[key])));featuregap=max(featuregap,gap);assert gap<1e-9
        for h,x in zip(hs,xx):
            p=expit(x@np.asarray(h['coefficients'])[:-1]+h['coefficients'][-1]);g=extra[extra.cutoff.eq(cutoff)&extra.seed.eq(seed)&extra.method.eq(h['method'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,rows);gap=float(np.max(np.abs(p-g.probability)));assert gap<1e-10;maxgap=max(maxgap,gap);np.testing.assert_array_equal(g.direction_up,(p>.5).astype(int))
        g=extra[extra.cutoff.eq(cutoff)&extra.seed.eq(seed)&extra.method.eq('native_mse')].sort_values('row_index');np.testing.assert_allclose(g.score,native,rtol=0,atol=1e-14);np.testing.assert_array_equal(g.direction_up,(native>0).astype(int));assert g.probability.isna().all()
        tr=training_rows(obs,cutoff);g=extra[extra.cutoff.eq(cutoff)&extra.method.eq('training_frequency')];np.testing.assert_allclose(g.probability,(targets['returns'][tr]>0).mean(),rtol=0,atol=1e-14)
        np.testing.assert_allclose(extra[extra.cutoff.eq(cutoff)].actual,targets['returns'][extra[extra.cutoff.eq(cutoff)].row_index],rtol=0,atol=1e-14)
        assert training.object_hash(model.state_dict())==r['model_sha256'];replays.append(dict(cutoff=cutoff,seed=seed,n=len(rows),training_unchanged=True,features_bitwise_equal=True,native_bitwise_equal=True));del model;torch.cuda.empty_cache()
    assert len(refs)==3*missing.cutoff.nunique() and len(extra)==16*len(missing)
    archive=archive_predictions();keys=['history','cutoff','row_index','method','seed'];assert not len(archive.merge(extra,on=keys));unique=pd.concat([archive,extra],ignore_index=True);pd.testing.assert_frame_equal(unique,csv('universal_model_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    feedback=independent_feedback(unique,obs,targets);pd.testing.assert_frame_equal(feedback,csv('feedback_pool'),check_dtype=False,atol=1e-14,rtol=0)
    events=csv('events');ledger=csv('published_ledger');members=csv('monitor_membership');audit=audit_policy(feedback,events,ledger,members);pd.testing.assert_frame_equal(audit,csv('independent_trigger_checks'),check_dtype=False)
    canonical=obs[obs.date.ge('2021-01-01')&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end'])];np.testing.assert_array_equal(ledger.row_index,canonical.index)
    candidate=pred[pred.history.eq('error16')].reset_index(drop=True);routed=route_predictions(unique,ledger);pd.testing.assert_frame_equal(candidate,routed,check_dtype=False,atol=1e-14,rtol=0)
    base,base_e=baselines();pd.testing.assert_frame_equal(pred[pred.history.ne('error16')].reset_index(drop=True),base,check_dtype=False,atol=1e-14,rtol=0);pd.testing.assert_frame_equal(ensemble[ensemble.history.ne('error16')].reset_index(drop=True),base_e,check_dtype=False,atol=1e-14,rtol=0)
    pd.testing.assert_frame_equal(ensemble_from(pred),ensemble,check_dtype=False,atol=1e-14,rtol=0)
    tables,pairs=evaluation.compute(pred,ensemble)
    for name,t in tables.items():pd.testing.assert_frame_equal(t,csv(name),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));assert len(pairs)==40
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['history','method','year']);independent.verify_metric_table(pred[pred.seed.ne(-1)],csv('seed_yearly_metrics'),['history','method','seed','year'])
    for w in evaluation.windows():
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['history','method']);independent.verify_metric_table(evaluation.select(pred[pred.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['history','method','seed'])
    # Counterfactual next-quarter diagnostics reconstructed directly from source rows.
    diag=csv('next_quarter_diagnostics')
    for row in diag.itertuples():
        index=cfg()['decision_dates'].index(row.cutoff);end=cfg()['decision_dates'][index+1] if index+1<len(cfg()['decision_dates']) else cfg()['label_end'];a=feedback[feedback.cutoff.eq(row.incumbent_cutoff)&feedback.date.gt(row.cutoff)&feedback.date.le(end)].sort_values('date');b=feedback[feedback.cutoff.eq(row.cutoff)&feedback.date.gt(row.cutoff)&feedback.date.le(end)].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==row.next_n
        loss_a=[(r.prediction_probability-r.actual_up)**2 for r in a.itertuples()];loss_b=[(r.prediction_probability-r.actual_up)**2 for r in b.itertuples()];assert abs(math.fsum(loss_a)/len(a)-row.incumbent_next_brier)<1e-14 and abs(math.fsum(loss_b)/len(b)-row.fresh_next_brier)<1e-14
        assert sum((r.prediction_probability>.5)==r.actual_up for r in a.itertuples())==row.incumbent_correct and sum((r.prediction_probability>.5)==r.actual_up for r in b.itertuples())==row.fresh_correct
    assert len(diag)==17 and len(pred)==21760 and len(ensemble)==8160;assert old_evidence()==prep['old_evidence'];check_frozen();pd.DataFrame(replays).to_csv(OUT/'extension_replays.csv',index=False)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=4600,coverage=coverage_result,reused_checkpoint_states=len(models),extension_replays=len(replays),new_neural_fits=0,new_head_fits=0,maximum_feature_algebra_gap=featuregap,maximum_probability_gap=maxgap,independent_trigger_checks=17,synthetic_cases=10,published_weekly_forecasts=272,independent_metrics=True,all40comparisons_recomputed=True,diagnostic_checks=17,archived_predictions_preserved=True,artifacts={'extension_replays.csv':sha(OUT/'extension_replays.csv')}));print('Verification PASS: chronological feedback, future perturbations, extensions, metrics and old evidence.',flush=True)
if __name__=='__main__':main()
