"""Development-only generation of R28 files, run before protocol freeze."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parents[1];OLD=P/'research_v27';NEW=P/'research_v28'
def write(n,s):(NEW/n).write_text(s,encoding='utf-8')
def adapted(name):return (OLD/f'{name}27.py').read_text(encoding='utf-8').replace('common27','common28').replace('contract27','contract28').replace('evaluate27','evaluate28')
obs=pd.read_csv(P/'research_v5/results/observation_table.csv');old=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
cfg={k:old[k] for k in ['label_end','windows','seeds','neural','probe','offset_probe','features','report_python']}
folds=[]
for cutoff in pd.date_range('2020-12-31','2026-06-30',freq='QE-DEC'):
    end=min(pd.Timestamp('2026-08-31'),pd.Timestamp(cutoff.year+1,12,31) if cutoff.month==12 else cutoff+pd.offsets.QuarterEnd())
    qend=min(pd.Timestamp('2026-08-31'),cutoff+pd.offsets.QuarterEnd())
    lower={f'rolling{k}':str((cutoff-pd.DateOffset(years=k)+pd.Timedelta(days=1)).date()) for k in [5,3]}
    mature=obs.joint_completed.le(str(cutoff.date()));tr={h:int((obs.date.ge(lo)&mature).sum()) for h,lo in lower.items()}
    mask=obs.date.gt(str(cutoff.date()))&obs.date.le(str(end.date()))&obs.weekday.eq(4)&obs.joint_completed.le(cfg['label_end'])
    folds.append(dict(cutoff=str(cutoff.date()),end=str(end.date()),quarter_end=str(qend.date()),lower=lower,train_n=int(mature.sum()),training_counts=tr,test_n=int(mask.sum())))
assert len(folds)==23
comparisons=[]
for h in ['rolling5','rolling3']:
    for change,candidate,reference in [('passes',h+'_annual20',h+'_matched'),('cadence',h+'_quarterly20',h+'_annual20')]:
        for method in old['primary_comparisons_per_window'][:0]:pass
        for method in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
            for loss in ['direction_error','brier']:comparisons.append(dict(intervention=change,history=candidate,candidate=method,reference_history=reference,reference=method,metric=loss))
    for policy in [h+'_annual20',h+'_quarterly20']:
        comparisons += [dict(intervention='control',history=policy,candidate='learned_vol_interaction',reference_history=policy,reference=m,metric=l) for m,l in [('training_frequency','brier'),('native_mse','direction_error')]]
assert len(comparisons)==32
feature_rows=sum(sum(f['training_counts'].values()) for f in folds)*3
steps=sum(sum((n+127)//128 for n in f['training_counts'].values()) for f in folds)*60
unique_rows=sum(f['test_n'] for f in folds)*2*16
cfg.update(version=28,experiment='Fixed 3/5-calendar-year windows, natural20 passes, annual versus quarterly WFO',
    folds=folds,memories=['rolling5','rolling3'],policies=['full','rolling5_matched','rolling3_matched','rolling5_annual20','rolling5_quarterly20','rolling3_annual20','rolling3_quarterly20'],
    scope='Two controlled policy axes: annual natural20 vs frozen R27 matched updates; quarterly natural20 vs annual natural20 within same window. Quarterly does not equalize annual compute. No cadence/window/epoch/seed selection after scoring.',
    training_membership='Trailing K calendar years of anchor dates: cutoff minus K years plus one calendar day; joint_completed<=cutoff. At year end identical rolling membership to R27.125bar input context may precede lower anchor boundary. Strict canonical data including original OHLC rounding tolerance0.011 unchanged.',
    training='Reset full network and optimizer from same seed at every distinct cutoff/window.20 permutations, each retained row once per epoch, no upsampling or early stopping. All target scales, neural weights, clipping, rays, projections and heads fitted on same mature retained rows. Not warm-start fine-tuning.',
    routing='Annual uses previous Dec31 cutoff all year; quarterly uses most recent calendar quarter-end strictly before signaldate.23 distinct cutoffs,6 shared yearends,17 extra. Models shared across policies at identical cutoff/window/seed; no duplicate fits.',
    inference_batching='Score union of dates needed by each distinct model once. December model union is next full year (2026 throughAug31); other cutoff union is nextquarter. Frozen feature extraction batches128; original native requires_grad=True flags and whole union as one batch. Route identical stored outputs to policies. No test-batch fitted transforms.',
    decisions='Mean probabilities of three fixed seeds, strictp>.5; native rawreturnmean>0 and no probability; frequency from identical mature training anchors. All six methods retained, all years and seeds reported.',
    sequencing='Freeze protocol/source/input/4002old files then causal contract. Complete all138 new backbones and552 heads before ANY new weekly scores. No calibration fits, no majority rule changes, no new feature/hyperparameter search.',
    primary_comparisons_per_window=comparisons,
    inference='64 contrasts jointlyHolm: two periods*(2windows*2interventions*3mainrecipes*2losses +4newpolicies*2matched controls). Circular8-observation10000bootstrap seed20260910 centeredtwo-sidedp,marginal95interval; negative loss difference better. Pooled272descriptive. Within-round adjustment cannot correct repeated historical exploration.',
    diagnostics='All methods, years, three seeds, reliability bins; seed probability variance identity and disagreement descriptive only. Cadence comparison reflects refreshing both data and parameters, not causal proof of a market regime. Total yearly compute differs; no fresh blind test.',
    verification='Independent datetime membership, calendar routing and natural permutation/batch counts; inherited raw input proof with frozen hashes. Replay138 checkpoint states/features, independently solve414 full convex heads and138 scalar heads, reconstruct forecasts, route and aggregate, independently verify all metrics; reproduce all64 contrasts; preserve old4002files.',
    limitations='All272 historical outcomes already viewed. Tests model age and data-window policy; cannot establish inherent necessity of rolling retraining or trading profitability. Quarter-end forecast labels near cutoff excluded until all targets mature. No automatic promotion.',
    budget=dict(new_neural_fits=138,shared_annual_neural_models=36,new_epochs=2760,new_optimizer_steps=steps,new_sample_presentations=feature_rows*20,new_unique_training_feature_rows=feature_rows,new_head_fits=552,new_projections=276,unique_model_prediction_rows=unique_rows,new_routed_model_prediction_rows=17408,reused_model_prediction_rows=13056,model_prediction_rows=30464,ensemble_rows=11424,weeks=272,old_files=4002))
write('protocol.json',json.dumps(cfg,indent=2,ensure_ascii=False))
s=adapted('train').replace('for history in NEW:', 'for history in MEMORIES:').replace("n=fold['train_n']","n=len(tr)")
s=s.replace('full_train_n=n','full_train_n=fold[\'train_n\']').replace('/36','/138').replace('/144','/552').replace('nominal epoch','natural epoch')
s=s.replace('len(models)==36 and len(heads)==144 and len(curves)==720 and sum(r[\'optimizer_steps\'] for r in curves)==17280',"len(models)==138 and len(heads)==552 and len(curves)==2760 and sum(r['optimizer_steps'] for r in curves)==cfg()['budget']['new_optimizer_steps']")
s=s.replace('neural_fits=36,nominal_epochs=720,optimizer_steps=17280,head_fits=144','neural_fits=138,natural_epochs=2760,optimizer_steps=cfg()[\'budget\'][\'new_optimizer_steps\'],head_fits=552').replace('projections=72','projections=276').replace('PASS:36backbones and144heads','PASS:138backbones and552heads')
write('train28.py',s)
s=adapted('score').replace("preds=[csv('full_model_predictions')]","preds=[]")
s=s.replace('models=pd.concat(preds,ignore_index=True);ensemble=ensemble_from(models);assert len(models)==13056 and len(ensemble)==4896',"unique=pd.concat(preds,ignore_index=True);assert len(unique)==cfg()['budget']['unique_model_prediction_rows'];unique.to_csv(OUT/'unique_model_predictions.csv',index=False)\n    models=pd.concat([csv('baseline_model_predictions'),route_predictions(unique)],ignore_index=True);ensemble=ensemble_from(models);assert len(models)==30464 and len(ensemble)==11424")
s=s.replace("ensemble[ensemble.history.eq('full')].reset_index(drop=True),csv('full_ensemble_predictions')","ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions')")
s=s.replace("['model_predictions.csv','ensemble_predictions.csv','testing_features.json']","['unique_model_predictions.csv','model_predictions.csv','ensemble_predictions.csv','testing_features.json']")
s=s.replace('new_model_rows=8704,reused_model_rows=4352,ensemble_rows=4896,new_neural_test_observations=1632',"new_model_rows=17408,reused_model_rows=13056,ensemble_rows=11424,unique_neural_test_observations=sum(f['test_n'] for f in cfg()['folds'])*6").replace('/36','/138')
write('score28.py',s)
s=adapted('evaluate').replace('len(pairs)==32','len(pairs)==64').replace('primary_contrasts=32,histories=3','primary_contrasts=64,histories=7')
write('evaluate28.py',s)
s=adapted('verify').replace("pred=csv('model_predictions')","pred=csv('unique_model_predictions')")
s=s.replace("schedule(len(tr),fold['train_n'],rng)","schedule(len(tr),len(tr),rng)").replace("row.presentations==fold['train_n']","row.presentations==len(tr)").replace("((fold['train_n']+127)//128)","((len(tr)+127)//128)").replace("row.optimizer_steps==(fold['train_n']+127)//128","row.optimizer_steps==(len(tr)+127)//128")
s=s.replace('/36','/138').replace('/144','/552')
s=s.replace("base,base_e=frozen_baselines();pd.testing.assert_frame_equal(pred[pred.history.eq('full')].reset_index(drop=True),base,check_dtype=False,atol=1e-14,rtol=0)","unique=pred;pred=csv('model_predictions');base,base_e=frozen_baselines();pd.testing.assert_frame_equal(pred[~pred.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,atol=1e-14,rtol=0)\n    routed=route_predictions(unique);pd.testing.assert_frame_equal(pred[pred.history.isin(NEW)].reset_index(drop=True),routed,check_dtype=False,atol=1e-14,rtol=0)\n    verify_routes(unique,pred,obs)")
a=s.index("    plan=csv('calibration_membership_plan')");b=s.index('    independent.verify_metric_table',a);s=s[:a]+s[b:]
s=s.replace('len(pred)==13056 and len(ensemble)==4896','len(pred)==30464 and len(ensemble)==11424').replace('len(solutions)==144 and len(replays)==36','len(solutions)==552 and len(replays)==138').replace('checkpoint_replays=36,independent_head_solutions=144','checkpoint_replays=138,independent_head_solutions=552').replace('testing_feature_rows=1632',"testing_feature_rows=sum(f['test_n'] for f in cfg()['folds'])*6")
s=s.replace('from contract28 import memberships','from contract28 import memberships,verify_routes')
write('verify28.py',s)
write('run28.py',adapted('run').replace("{phase}27.py","{phase}28.py"))
write('delivery28.py',adapted('delivery'))
print(json.dumps(cfg['budget'],indent=2));print('Files generated; not frozen, no fits or new scores.')
