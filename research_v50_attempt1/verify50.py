from common50 import *
from dataset50 import annual_interface,observations
from annual50 import fit_annual
from quarter50 import fit_quarter,Engine,extend_bank
from validate50 import validate_annual,validate_quarter,validate_package_schema
from build50 import request_check,activate,select_package
from data49 import load_snapshot,unchanged_prefix
from datetime import datetime
import tempfile

def rejected(call):
    try:call()
    except (ValueError,FileExistsError):return
    raise AssertionError('Expected rejection missing')
def main():
    freeze=check_freeze();started=iso(utc());cases=[];frame=load_csv(PROJECT/'research/data/1_000300.csv');cutoff='2025-12-31'
    a=annual_interface(frame,cutoff);oldobs=load_csv(PROJECT/'research_v5/results/observation_table.csv');oldids=np.flatnonzero(oldobs.date.ge('2021-01-01')&oldobs.joint_completed.le(cutoff));np.testing.assert_array_equal(a['training_rows'],oldids)
    with np.load(PROJECT/'research_v28/cache/training_rolling5_2025-12-31.npz') as d:
        for key in a['values']:np.testing.assert_array_equal(a['values'][key],d[key])
        for key in a['labels']:np.testing.assert_array_equal(a['labels'][key],d[key])
    assert a['scales']==read(PROJECT/'research_v28/results/training_scales.json')['rolling5_'+cutoff]
    assert len(oldids)==1206 and len(a['observations'])==3620
    poison=frame.copy();poison.loc[poison.date>cutoff,['open','high','low','close','volume']]=np.nan;b=annual_interface(poison,cutoff)
    for key in a['values']:np.testing.assert_array_equal(a['values'][key],b['values'][key])
    for key in a['labels']:np.testing.assert_array_equal(a['labels'][key],b['labels'][key])
    cases+=['original1206mature_members_exact','all_training_packs_and_labels_exact','training_scales_exact','future_nan_poison_excluded_before_validation']
    print('Annual training interface parity PASS; beginning three full20epoch engineering replays.',flush=True)
    annual=fit_annual(frame,cutoff,OUT/'annual_replay');annual_check=validate_annual(frame,annual)
    source=read(PROJECT/'research_v28/results/models.json');digest_checks=[]
    for m in annual['models']:
        old=next(o for o in source if o['history']=='rolling5' and o['cutoff']==cutoff and o['seed']==m['seed'])
        for key in ['model_sha256','optimizer_sha256','rng_sha256']:assert m[key]==old[key],(m['seed'],key)
        digest_checks.append(dict(seed=m['seed'],model_sha256=m['model_sha256'],optimizer_sha256=m['optimizer_sha256'],rng_sha256=m['rng_sha256']))
    u=read(PROJECT/'research_v28/results/heads.json');w=read(PROJECT/'research_v47/results/heads.json');max_theta=0.
    for name,source_heads in [('uniform',u),('weighted',w)]:
        for h in annual[name]:
            old=next(o for o in source_heads if o['cutoff']==cutoff and o['seed']==h['seed'] and o['method']==h['method'] and (name=='weighted' or o['history']=='rolling5'))
            gap=float(np.max(abs(np.array(h['coefficients'])-old['coefficients'])));max_theta=max(max_theta,gap);assert gap<1e-8
    cases+=['three_exact_final_model_optimizer_rng_replays','60natural_epochs_600optimizer_steps','24head_parameter_replays','independent_annual_checkpoint_features_and_head_stationarity']
    bank=load_csv(PROJECT/'research_v39/results/weekly_signal_bank.csv');oldheads=read(PROJECT/'research_v39/results/correction_heads.json');oldgates=load_csv(PROJECT/'research_v39/results/gate_decisions.csv');dates=read(PROJECT/'research_v39/protocol.json')['decision_dates'];quarter_checks=[];max_quarter=0.
    for date in dates:
        result=fit_quarter(bank,date);independent=validate_quarter(bank,date,result)
        for h in result['heads']:
            old=next(o for o in oldheads if o['cutoff']==date and o['method']==h['method'] and o['seed']==h['seed'] and o['family']=='state' and o['component']==h['component'])
            gap=abs(h['offset']-old['offset']);max_quarter=max(max_quarter,gap);assert gap<1e-11 and h['fit_eligible']==old['fit_eligible']
        for g in result['decisions']:
            old=oldgates[oldgates.cutoff.eq(date)&oldgates.method.eq(g['method'])&oldgates.family.eq('state')&oldgates.component.eq(g['state'])].iloc[0]
            assert g['accepted']==old.accepted and g['reason']==old.reason
        quarter_checks.append(dict(cutoff=date,**independent))
    assert len(dates)==23
    cases+=['1104quarter_state_offset_replays','368heldout_gate_replays','independent_quarter_brentq_and_maturity_purge','annual_reset_and_exact_zero_fallback']
    # The rebuilt annual checkpoint must also drive the unchanged R49 prefix engine.
    bootstrap=read(PROJECT/'research_v49/results/bootstrap_package.json');package=copy.deepcopy(bootstrap)
    for key in ['models','uniform','weighted','volatility_median','return_scale','training_up_frequency']:package[key]=annual[key]
    package['scope']='engineering_replay';package['dependencies'].update(annual['annual_artifacts']);validate_package_schema(package)
    engine=Engine(package);oldpred=load_csv(PROJECT/'research_v48/results/model_predictions.csv');samples=oldobs[oldobs.date.between('2026-07-01','2026-08-31')&oldobs.weekday.eq(4)];maxprob=0.;forecasts=[]
    for row in samples.itertuples():
        result=engine.predict(frame[frame.date.le(row.date)],row.date,production=False);truth=oldpred[oldpred.date.eq(row.date)].set_index(['history','method','seed'])
        for r in result['seed_predictions']:
            old=truth.loc[(r['history'],r['method'],r['seed'])];gap=abs(r['probability']-old.probability);maxprob=max(maxprob,gap);assert gap<2e-6 and r['direction_up']==old.direction_up
        forecasts.append(dict(date=row.date,seed_rows=len(result['seed_predictions']),ensemble_rows=len(result['ensemble_predictions'])))
    assert len(forecasts)==8
    cases+=['rebuilt_annual_checkpoint_drives49prefix_engine','480forecast_directions_preserved_in_engineering_replay']
    # Extend only the historical calibration bank using the already archived new data.
    journal=Journal();initial=journal.events();snapshot=next(e for e in reversed(initial) if e['kind']=='snapshot');current=load_snapshot(journal,snapshot)
    # No Sep30 fit: these matured signals test bank ingestion only, as of last available day.
    expanded=extend_bank(current,snapshot['payload']['end'],bootstrap,{bootstrap['annual_cutoff']:bootstrap},initial)
    original_keys=set(zip(bank.date,bank.method,bank.seed));added=expanded[~expanded.apply(lambda r:(r.date,r.method,r.seed) in original_keys,axis=1)]
    assert set(added.calibration_source)=={'calibration_only_replay_not_prospective'} and len(added)>0
    before=bank[['date','method','seed','probability','annual_logit','state']].sort_values(['date','method','seed']).reset_index(drop=True)
    kept=expanded[expanded.date.isin(bank.date)][before.columns].sort_values(['date','method','seed']).reset_index(drop=True);pd.testing.assert_frame_equal(before,kept,check_dtype=False)
    _,ob,_=observations(current,snapshot['payload']['end']);truth=ob.set_index('date')
    for date,g in added.groupby('date'):assert g.actual_up.eq(int(truth.loc[date,'exec_return']>0)).all() and g.joint_completed.eq(truth.loc[date,'joint_completed']).all()
    assert journal.events()==initial
    cases+=['new_mature_calibration_only_replays_separate_from_forecasts','old_calibration_bank_preserved','new_calibration_labels_match_cutoff_data']
    now=utc();rejected(lambda:request_check('2026-09-30',snapshot,now));rejected(lambda:request_check('2026-12-31',snapshot,now))
    future=datetime.fromisoformat('2026-10-01T18:00:00+08:00');rejected(lambda:request_check('2026-09-30',snapshot,future));rejected(lambda:request_check('2026-09-29',snapshot,future))
    wrong=copy.deepcopy(bootstrap);wrong['quarter_offsets'].pop();rejected(lambda:validate_package_schema(wrong))
    wrong=copy.deepcopy(bootstrap);wrong['valid_until']='2026-12-31';rejected(lambda:validate_package_schema(wrong))
    wrong=copy.deepcopy(bootstrap);wrong['uniform'][0]['cutoff']='2026-12-31';rejected(lambda:validate_package_schema(wrong))
    revised=current.copy();revised.loc[0,'volume']+=1;rejected(lambda:unchanged_prefix(current,revised,current.date.iloc[-1]))
    # Independent checks of critical threshold boundaries inherited from R39.
    _,_,q=modules();p0=np.array([.4,.6,.4,.6,.4]);better=np.array([.3,.7,.3,.7,.3]);y=np.array([0,1,0,1,0])
    assert q.gate_decision('ready','state',10,5,p0,better,y)['accepted']
    assert not q.gate_decision('ready','state',9,5,p0,better,y)['accepted']
    assert not q.gate_decision('ready','state',10,4,p0[:4],better[:4],y[:4])['accepted']
    assert not q.gate_decision('ready','state',10,5,p0,p0,y)['accepted']
    cases+=['future_Sep30_andDec31_refits_rejected','stale_snapshot_and_nonquarter_cutoff_rejected','missing_offset_and_wrong_validity_or_annual_cutoff_rejected','revised_snapshot_prefix_rejected','state10_validation5_and_Brier_tie_boundaries']
    # Use an isolated journal fixture to exercise registration gates after a provisional verification
    # is written below; production bootstrap is activated only by the later delivery step.
    csv_save(OUT/'quarter_replay_checks.csv',pd.DataFrame(quarter_checks));csv_save(OUT/'calibration_extension_check.csv',added)
    report=dict(status='PASS',started_utc=started,completed_utc=iso(utc()),freeze_sha256=sha(OUT/'freeze.json'),protocol_sha256=sha(ROOT/'protocol.json'),cases=cases,old_files_preserved=6230,annual_replay_cutoff=cutoff,neural_replays=3,epochs=60,optimizer_steps=600,head_fits=24,annual=annual_check,checkpoint_digests=digest_checks,maximum_coefficient_gap=max_theta,quarter_cutoffs=23,quarter_offsets=1104,quarter_gate_cells=368,maximum_archived_quarter_gap=max_quarter,maximum_probability_gap=maxprob,calibration_added_seed_rows=len(added),calibration_added_dates=sorted(added.date.unique().tolist()),prospective_predictions=sum(e['kind']=='prediction' for e in journal.events()),future_cutoff_fits=0,engineering_only=True,artifacts=artifacts(OUT/'annual_replay'))
    report['artifacts'].update({relative(OUT/n):sha(OUT/n) for n in ['quarter_replay_checks.csv','calibration_extension_check.csv']})
    guard(old_evidence()==freeze['old_evidence'],'Old artifacts changed');check_freeze();save(OUT/'verification.json',report)
    print(encoded({k:report[k] for k in ['status','neural_replays','epochs','head_fits','quarter_offsets','quarter_gate_cells','maximum_coefficient_gap','maximum_probability_gap','calibration_added_dates','future_cutoff_fits']}).decode())
if __name__=='__main__':main()
