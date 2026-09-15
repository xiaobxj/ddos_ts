from common31 import *
from contract31 import coverage_checks,synthetic_cases,audit_policy
import evaluate31 as evaluation
import verify15 as independent
import math

def independent_feedback(unique,obs,targets):
    rows=[]
    for (cutoff,i),g in unique.groupby(['cutoff','row_index'],sort=True):
        r=g[g.method.eq('learned_vol_interaction')].sort_values('seed');assert r.seed.tolist()==cfg()['seeds'];freq=int((targets['returns'][training_rows(obs,cutoff)]>0).sum())/len(training_rows(obs,cutoff));assert abs(freq-g[g.method.eq('training_frequency')].probability.iloc[0])<1e-14
        rows.append(dict(row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],cutoff=cutoff,prediction_probability=math.fsum(r.probability)/3,frequency_probability=freq,actual_up=int(targets['returns'][i]>0)))
    return pd.DataFrame(rows,columns=previous_round.FEEDBACK_COLUMNS)

def independent_blend(base,actual,route):
    lookup={(r.history,r.method,r.seed,r.date):r for r in base.itertuples()};routes=route.set_index('date');gap=0.
    for r in actual.itertuples():
        dates=[d for d in cfg()['decision_dates'] if d<r.date];quarter=max(dates);annual=max(d for d in dates if d.endswith('12-31'));assert r.cutoff==quarter
        a=lookup[('rolling5_annual20',r.method,r.seed,r.date)];b=lookup[('rolling5_quarterly20',r.method,r.seed,r.date)];assert a.cutoff==annual and b.cutoff==quarter
        rr=routes.loc[r.date];assert rr.annual_cutoff==annual and rr.quarterly_cutoff==quarter and rr.annual_weight==rr.quarterly_weight==.5
        s=math.fsum([a.score,b.score])/2;gap=max(gap,abs(s-r.score));assert abs(s-r.score)<1e-14
        if r.method=='native_mse':assert pd.isna(r.probability) and r.direction_up==int(s>0)
        else:
            p=math.fsum([a.probability,b.probability])/2;gap=max(gap,abs(p-r.probability));assert abs(p-r.probability)<1e-14 and r.direction_up==int(p>.5)
    assert len(actual)==4352 and len(route)==272;return gap

def independent_inference(ensemble,pairs):
    ps=[]
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(0,n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);assert np.array_equal(ids,statistics.bootstrap_indices(n))
        for row in [r for r in pairs if r['window']==w['name']]:
            subset=ensemble[ensemble.date.between(w['start'],w['end'])];a=subset[subset.history.eq(row['history'])&subset.method.eq(row['candidate'])].sort_values('date');b=subset[subset.history.eq(row['reference_history'])&subset.method.eq(row['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==n
            if row['metric']=='direction_error':values=(a.direction_up.to_numpy()!=a.actual_up.to_numpy()).astype(float)-(b.direction_up.to_numpy()!=b.actual_up.to_numpy()).astype(float)
            else:values=np.square(a.probability.to_numpy()-a.actual_up.to_numpy())-np.square(b.probability.to_numpy()-b.actual_up.to_numpy())
            mean=float(values.mean());samples=values[ids].mean(axis=1);center=(values-mean)[ids].mean(axis=1);p=(1+int((np.abs(center)>=abs(mean)).sum()))/10001;lo,hi=np.quantile(samples,[.025,.975]);assert abs(mean-row['difference'])<1e-14 and abs(lo-row['ci95_low'])<1e-14 and abs(hi-row['ci95_high'])<1e-14 and p==row['p'];ps.append(p)
    assert ps==[r['p'] for r in pairs];order=sorted(range(len(ps)),key=lambda i:ps[i]);adjusted=[0.]*len(ps);last=0.
    for rank,i in enumerate(order):last=max(last,min(1.,ps[i]*(len(ps)-rank)));adjusted[i]=last
    np.testing.assert_array_equal(adjusted,[r['holm_adjusted_p'] for r in pairs])

def main():
    started=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['source_sha256','protocol_sha256','input_sha256']:assert r[key]==prep[key]
    coverage=coverage_checks();assert len(synthetic_cases())==16;policy=read(OUT/'policy_manifest.json');assert check_phase('scoring')['started_utc']<=policy['completed_utc']<=check_phase('scoring')['finished_utc']
    for n,d in policy['artifacts'].items():assert sha(ROOT/n)==d
    obs,_,targets=data();scales=read(V28/'results/training_scales.json');models=read(OUT/'source_models.json')
    for r in models:
        s=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True);assert s['protocol_sha256']==sha(V28/'protocol.json') and s['history']=='rolling5' and s['epoch']==20 and s['cutoff']==r['cutoff'] and s['seed']==r['seed'];assert training.object_hash(s['state_dict'])==r['model_sha256'] and training.object_hash(s['optimizer_state_dict'])==r['optimizer_sha256'];assert s['scales']==scales['rolling5_'+r['cutoff']]
    unique=csv('universal_model_predictions');np.testing.assert_allclose(unique.actual,targets['returns'][unique.row_index],rtol=0,atol=1e-14);assert unique.date.tolist()==obs.date.iloc[unique.row_index].tolist() and unique.joint_completed.tolist()==obs.joint_completed.iloc[unique.row_index].tolist()
    feedback=independent_feedback(unique,obs,targets);pd.testing.assert_frame_equal(feedback,csv('feedback_pool'),check_dtype=False,atol=1e-14,rtol=0)
    events=csv('events');ledger=csv('paired_ledger');members=csv('trial_membership');audit=audit_policy(feedback,events,ledger,members);pd.testing.assert_frame_equal(audit,csv('independent_trial_checks'),check_dtype=False);assert events.ready.sum()==11 and events.reason.eq('same_model').sum()==6
    canonical=obs[obs.date.ge('2021-01-01')&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end'])];np.testing.assert_array_equal(ledger.row_index,canonical.index);assert ledger.date.is_unique
    route=csv('shadow_routing');pd.testing.assert_frame_equal(shadow_route(ledger),route,check_dtype=False);pred=csv('model_predictions');ensemble=csv('ensemble_predictions');base,be=baselines()
    pd.testing.assert_frame_equal(pred[~pred.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    pd.testing.assert_frame_equal(pred[pred.history.eq('shadow_trial')].reset_index(drop=True),route_predictions(unique,route),check_dtype=False,check_exact=True)
    gap=independent_blend(base,pred[pred.history.eq('annual_quarter_half')],csv('blend_routing'));pd.testing.assert_frame_equal(ensemble_from(pred),ensemble,check_dtype=False,atol=1e-14,rtol=0)
    tables,pairs=evaluation.compute(pred,ensemble)
    for name,t in tables.items():pd.testing.assert_frame_equal(t,csv(name),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));assert len(pairs)==68;independent_inference(ensemble,pairs)
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['history','method','year']);independent.verify_metric_table(pred[pred.seed.ne(-1)],csv('seed_yearly_metrics'),['history','method','seed','year'])
    for w in evaluation.windows():
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['history','method']);independent.verify_metric_table(evaluation.select(pred[pred.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['history','method','seed'])
    diag=csv('next_quarter_diagnostics');assert len(diag)==11
    for r in diag.itertuples():
        i=cfg()['decision_dates'].index(r.cutoff);end=cfg()['decision_dates'][i+1] if i+1<len(cfg()['decision_dates']) else cfg()['label_end'];a=feedback[feedback.cutoff.eq(r.incumbent_cutoff)&feedback.date.gt(r.cutoff)&feedback.date.le(end)].sort_values('date');b=feedback[feedback.cutoff.eq(r.challenger_cutoff)&feedback.date.gt(r.cutoff)&feedback.date.le(end)].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==r.next_n
        gain=sum((x.prediction_probability>.5)==x.actual_up for x in b.itertuples())-sum((x.prediction_probability>.5)==x.actual_up for x in a.itertuples());assert gain==r.challenger_correct-r.incumbent_correct;d=math.fsum((bb.prediction_probability-aa.actual_up)**2-(aa.prediction_probability-aa.actual_up)**2 for aa,bb in zip(a.itertuples(),b.itertuples()))/len(a);assert abs(d-r.challenger_minus_incumbent_brier)<1e-14
    assert len(pred)==30464 and len(ensemble)==11424;pd.testing.assert_frame_equal(runtime_budgets(events),csv('budgets'),check_dtype=False);assert old_evidence()==prep['old_evidence'];check_frozen()
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=4696,coverage=coverage,reused_checkpoint_states=69,new_neural_fits=0,new_head_fits=0,new_inference_passes=0,independent_trial_checks=17,synthetic_cases=16,paired_weekly_forecasts=272,distinct_trials=11,maximum_blend_algebra_gap=gap,independent_metrics=True,all68comparisons_recomputed=True,independent_bootstrap_and_holm=True,following_quarter_diagnostics_checked=11,archived_predictions_preserved=True));print('Verification PASS: independent chronological trials, blending, 68 comparisons and old evidence.',flush=True)
if __name__=='__main__':main()
