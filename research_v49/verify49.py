"""Engineering replay and adversarial contract tests, never prospective scores."""
from common49 import *
from data49 import clean,prefix,label_for,load_snapshot,unchanged_prefix
from forecast49 import Engine,inputs,package_check,validate_forecast
from evaluate49 import summarize,comparison
from run49 import record,status
import tempfile,copy,unittest.mock

def rejected(call):
    try:call()
    except (ValueError,FileExistsError):return
    raise AssertionError('Required rejection did not occur')
def fixture_forecast(p):
    return dict(state=STATES[0],seed_predictions=[dict(history=h,method=m,seed=s,probability=p,direction_up=int(p>.5),logit=0.) for s in SEEDS for m in METHODS for h in HISTORIES],ensemble_predictions=[dict(history=h,method=m,probability=float(pd.Series([p,p,p]).mean()),direction_up=int(p>.5)) for h in HISTORIES for m in METHODS])
def main():
    check_freeze();started=iso(utc());cases=[];package=read(OUT/'bootstrap_package.json')
    frame=clean(pd.read_csv(PROJECT/'research/data/1_000300.csv',float_precision='round_trip'))
    obs=pd.read_csv(PROJECT/'research_v5/results/observation_table.csv',float_precision='round_trip')
    models=pd.read_csv(PROJECT/'research_v48/results/model_predictions.csv',float_precision='round_trip')
    ensembles=pd.read_csv(PROJECT/'research_v48/results/ensemble_predictions.csv',float_precision='round_trip')
    samples=obs[obs.date.between('2026-07-01','2026-08-31')&obs.weekday.eq(4)]
    guard(len(samples)==8,'Replay sample count changed');engine=Engine(package);gaps=[];native_gaps=[];feature_gaps=[];replay=[]
    refs=read(PROJECT/'research_v48/results/source_testing_features.json')
    cache={}
    for seed in SEEDS:
        ref=next(r for r in refs if r['cutoff']=='2025-12-31' and r['seed']==seed)
        with np.load(PROJECT/ref['cache_file']) as d:cache[seed]={k:d[k].copy() for k in d.files}
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as data:packed={k:data[k][samples.index.to_numpy()].copy() for k in ['patches','geometry','valid']}
    for j,(idx,row) in enumerate(samples.iterrows()):
        observed=frame[frame.date<=row.date].reset_index(drop=True);_,v=inputs(observed,row.date)
        for k in v:np.testing.assert_array_equal(v[k][0],packed[k][j])
        result=engine.predict(observed,row.date,production=False)
        old=models[models.date.eq(row.date)&models.history.isin(HISTORIES)].set_index(['history','method','seed'])
        for r in result['seed_predictions']:
            o=old.loc[(r['history'],r['method'],r['seed'])];gap=abs(r['probability']-o.probability);gaps.append(gap)
            assert gap<2e-6 and r['direction_up']==o.direction_up
            replay.append(dict(date=row.date,history=r['history'],method=r['method'],seed=r['seed'],new_probability=r['probability'],archived_probability=o.probability,absolute_gap=gap))
        for r in result['ensemble_predictions']:
            o=ensembles[ensembles.date.eq(row.date)&ensembles.history.eq(r['history'])&ensembles.method.eq(r['method'])].iloc[0]
            assert abs(r['probability']-o.probability)<2e-6 and r['direction_up']==o.direction_up
        for d in result['inference_diagnostics']:
            pos=np.flatnonzero(cache[d['seed']]['row_index']==idx)[0]
            gap=float(np.max(abs(np.array(d['features'])-cache[d['seed']]['features'][pos])));feature_gaps.append(gap);assert gap<2e-5
        for seed,native in result['controls']['native_returns_by_seed'].items():
            gap=abs(native-old.loc[(U,'native_mse',int(seed)),'score']);native_gaps.append(gap);assert gap<2e-6
        assert result['controls']['training_frequency']==old.loc[(U,'training_frequency',-1),'probability']
        print(f'Engineering replay {j+1}/8: {row.date}; no prospective record.',flush=True)
    cases+=['raw125_transform_and25x5packing_exact_parity','480learned_seed_and160ensemble_predictions_direction_parity','single_sample_GPU_roundoff_bounded','native_and_frequency_control_parity']
    # Complete label formula check, including holiday-length weeks.
    labels=obs[obs.date.ge('2021-01-01')&obs.weekday.eq(4)]
    for r in labels.itertuples():
        actual=label_for(frame,r.date);assert actual['status']=='MATURE' and actual['joint_completed']==r.joint_completed and actual['exit_date']==r.exit_date and abs(actual['exec_return']-r.exec_return)<1e-14
    assert len(labels)==272 and (labels.sessions!=5).any()
    for end in ['2026-08-21','2026-08-28']:assert label_for(frame[frame.date<=end],'2026-08-21')['status']=='PENDING'
    bad=frame.copy();k=bad.index[bad.date.eq('2026-08-24')][0];bad.loc[k,'high']=bad.loc[k,'low']*.5
    assert label_for(bad,'2026-08-21')['status']=='INVALID'
    flat=frame.copy();flat.loc[flat.date.eq('2026-08-31'),'open']=float(flat.loc[flat.date.eq('2026-08-24'),'open'].iloc[0]);flat.loc[flat.date.eq('2026-08-31'),'high']=flat.loc[flat.date.eq('2026-08-31'),['open','high','close']].max(axis=1)
    assert label_for(flat,'2026-08-21')['actual_up']==0
    cases+=['272original_joint_labels_and_holiday_maturities','label_waits_for_exit_and5auxiliary_bars','future_invalid_OHLC_excluded','zero_return_is_not_up']
    observed=frame[frame.date<='2026-08-21'].copy();poison=frame.copy();rejected(lambda:inputs(poison,'2026-08-21'))
    invalid=observed.copy();invalid.iloc[-1,invalid.columns.get_loc('high')]=1.;rejected(lambda:inputs(invalid,'2026-08-21'))
    rejected(lambda:clean(pd.concat([observed,observed.tail(1)])))
    rejected(lambda:inputs(observed.tail(124),'2026-08-21'))
    unchanged_prefix(observed,frame,'2026-08-21')
    revised=frame.copy();revised.loc[0,'volume']+=1;rejected(lambda:unchanged_prefix(observed,revised,'2026-08-21'))
    rejected(lambda:unchanged_prefix(observed,frame.iloc[1:],'2026-08-21'))
    rejected(lambda:package_check(package,'2026-10-09',utc(),False));rejected(lambda:package_check(package,'2027-01-08',utc(),False))
    p=fixture_forecast(.5);validate_forecast(p);assert all(r['direction_up']==0 for r in p['ensemble_predictions'])
    missing=copy.deepcopy(p);missing['seed_predictions'].pop();rejected(lambda:validate_forecast(missing))
    invalid=copy.deepcopy(p);invalid['seed_predictions'][0]['probability']=float('nan');rejected(lambda:validate_forecast(invalid))
    mixed=copy.deepcopy(p);mixed['ensemble_predictions'][0]['direction_up']=1;rejected(lambda:validate_forecast(mixed))
    cases+=['future_rows_rejected_at_input_interface','invalid125bar_prefix_rejected','duplicate_dates_and_short_inputs_rejected','revision_and_omitted_prefix_rejected','quarter_and_annual_expiration_rejected','strict_point5_boundary','missing_seed_nonfinite_and_changed_direction_rejected']
    date=cfg()['signal_slots'][0]
    assert not record_window(date,datetime.fromisoformat(date+'T17:59:59+08:00'))
    assert record_window(date,datetime.fromisoformat(date+'T18:00:00+08:00'))
    assert not record_window(date,datetime.fromisoformat(date+'T23:00:00+08:00'))
    assert not record_window(date,datetime.fromisoformat(date+'T18:00:00+08:00')+timedelta(days=1))
    journal=Journal();before=journal.events();rejected(lambda:record(journal));assert journal.events()==before==[]
    with tempfile.TemporaryDirectory(prefix='r49_contract_') as directory:
        temp=Journal(Path(directory)/'synthetic');temp.initialize('synthetic');blob,h=temp.blob(b'fixture only','.txt')
        temp.append('diagnostic','one',dict(note='synthetic fixture'),{blob:h});temp.events()
        rejected(lambda:temp.append('diagnostic','one',{}));rejected(lambda:temp.append('label',date,{}))
        rejected(lambda:temp.append('prediction',date,{}))
        with temp.locked():rejected(lambda:temp.append('diagnostic','two',{}))
        (temp.root/'blobs'/blob).write_bytes(b'corrupted fixture');rejected(temp.events)
    with tempfile.TemporaryDirectory(prefix='r49_chain_') as directory:
        temp=Journal(Path(directory)/'synthetic');temp.initialize('synthetic');temp.append('diagnostic','one',{});temp.append('diagnostic','two',{})
        path=temp.root/'events/000001.json';e=read(path);e['payload']={'tampered':True};path.write_bytes(encoded(e));rejected(temp.events)
    cases+=['Friday_18inclusive23exclusive_and_same_day_guard','backfill_command_rejected_with_empty_real_journal','duplicate_and_concurrent_writers_rejected','orphan_label_and_synthetic_prediction_rejected','snapshot_blob_and_journal_chain_tamper_detected']
    # All evaluation objects below are memory-only synthetic fixtures; no journal writes.
    events=[]
    for j,date in enumerate(cfg()['signal_slots']):
        if j in [3,10,19,30,45,50]:continue
        events += [dict(kind='prediction',key=date,payload={'forecast':fixture_forecast(.6 if j%2 else .4)}),dict(kind='label',key=date,payload=dict(status='MATURE',actual_up=j%2))]
    early=summarize(events,datetime.fromisoformat(cfg()['signal_slots'][12]+'T23:01:00+08:00'));assert early['comparisons']==[]
    review=datetime.fromisoformat(cfg()['cohort']['primary_review_not_before']+'T23:01:00+08:00')
    final=summarize(events,review);assert final['status']=='PRIMARY_READY' and len(final['comparisons'])==42 and final['complete_mature_signals']==46
    assert all(r['p']==r['holm_adjusted_p']==1 for r in final['comparisons'])
    assert sum(r['status']=='NO_TIMELY_PREDICTION' for r in final['coverage'])==6
    assert summarize(events[:60],review)['comparisons']==[]
    pending=events[:-1];assert summarize(pending,review)['comparisons']==[]
    v=np.tile([.1,-.05,.02,-.03],13).astype(float);v[[3,10,19,30,45,50]]=np.nan
    a=comparison(v,cfg()['inference']);assert a==comparison(v,cfg()['inference']) and a['n']==46
    rng=np.random.default_rng(20260910);starts=rng.integers(52,size=(10000,7));means=[]
    for starts_row in starts:
        ids=[(int(s)+offset)%52 for s in starts_row for offset in range(8)][:52];values=[v[i] for i in ids if np.isfinite(v[i])];means.append(sum(values)/len(values))
    mu=float(np.nanmean(v));np.testing.assert_allclose([a['difference'],a['ci95_low'],a['ci95_high']],[mu,*np.quantile(means,[.025,.975])],rtol=0,atol=1e-14)
    expected_p=(1+sum(abs(x-mu)>=abs(mu) for x in means))/10001;assert abs(a['p']-expected_p)<=1/10001
    cases+=['no_primary_testing_before_fixed_review','minimum40_and_pending_label_gates','all52slots_visible_no_missing_slot_replacement','42comparison_family_Holm_and_zero_difference','calendar_gap_preserving_bootstrap_independent_replay']
    check_freeze();assert old_evidence()==read(OUT/'freeze.json')['old_evidence'];assert journal.events()==[]
    replay_path=OUT/'engineering_replay.csv';exclusive(replay_path,pd.DataFrame(replay).to_csv(index=False,lineterminator='\n').encode('utf-8'))
    save(OUT/'verification.json',dict(status='PASS',started_utc=started,completed_utc=iso(utc()),cases=cases,old_files_preserved=6206,replayed_fridays=8,new_historical_inference_seed_windows=24,learned_seed_predictions_checked=len(replay),ensemble_predictions_checked=160,label_parity_rows=272,maximum_probability_gap=max(gaps),maximum_feature_gap=max(feature_gaps),maximum_native_return_gap=max(native_gaps),new_neural_fits=0,new_head_fits=0,real_prospective_predictions=0,synthetic_results_are_not_prospective=True,artifacts={str(replay_path.relative_to(ROOT)):sha(replay_path)},protocol_sha256=sha(ROOT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json')))
    print(json.dumps(dict(status='PASS',checks=len(cases),maximum_probability_gap=max(gaps),real_prospective_predictions=0),indent=2))
if __name__=='__main__':main()
