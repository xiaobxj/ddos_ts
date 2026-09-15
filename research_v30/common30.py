"""Causal persistent forecast-underperformance trigger; immutable R28 weights."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V29=PROJECT/'research_v29';V28=PROJECT/'research_v28'
sys.path.insert(0,str(V29));import common29 as previous_round
prior=previous_round.prior
np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;training=prior.training;neural=prior.neural
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;design=prior.design;probability=prior.probability
data=prior.data;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates;load_model=prior.load_model;forecasts=prior.forecasts;apply_pipeline=prior.apply_pipeline
statistics=prior.statistics;METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
HISTORIES=['rolling5_matched','rolling5_annual20','rolling5_quarterly20','state90','error16'];NEW=['error16']
fold_for=previous_round.fold_for;training_rows=previous_round.training_rows;predict_rows=previous_round.predict_rows;ensemble_from=previous_round.ensemble_from
FEEDBACK_COLUMNS=['row_index','date','joint_completed','cutoff','prediction_probability','frequency_probability','actual_up']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V29/'protocol.json']+[V29/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','extension_predictions.csv','model_predictions.csv','ensemble_predictions.csv','budgets.csv']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V29/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V29/'results/delivery_manifest.json')['files'].items():r[str((V29/n).relative_to(PROJECT))]=d
    p=V29/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4600
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    p=read(OUT/f'{phase}_manifest.json');assert p.get('finished_utc')
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def archive_predictions():
    a=previous_round.archive_predictions();b=pd.read_csv(V29/'results/extension_predictions.csv',float_precision='round_trip');r=pd.concat([a,b],ignore_index=True);assert not r.duplicated(['cutoff','method','seed','date']).any();return r
def baselines():
    m=pd.read_csv(V29/'results/model_predictions.csv',float_precision='round_trip');e=pd.read_csv(V29/'results/ensemble_predictions.csv',float_precision='round_trip');return m,e
def coverage(obs):
    rows=[]
    for cutoff in cfg()['decision_dates']:
        end=min(cfg()['label_end'],str(int(cutoff[:4])+1 if cutoff.endswith('12-31') else int(cutoff[:4]))+'-12-31')
        ids=np.flatnonzero(obs.date.gt(cutoff)&obs.date.le(end)&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']))
        for i in ids:rows.append(dict(cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
    return pd.DataFrame(rows)
def extensions(cover):
    a=pd.read_csv(V28/'results/unique_model_predictions.csv',usecols=['history','cutoff','row_index']);a=a[a.history.eq('rolling5')];b=pd.read_csv(V29/'results/extension_predictions.csv',usecols=['cutoff','row_index']);keys=set(zip(a.cutoff,a.row_index))|set(zip(b.cutoff,b.row_index))
    return cover[[(c,int(i)) not in keys for c,i in zip(cover.cutoff,cover.row_index)]].reset_index(drop=True)
def feedback_from(unique):
    keys=['cutoff','row_index','date','joint_completed'];r=unique[unique.method.eq('learned_vol_interaction')]
    groups=r.groupby(keys,sort=True);assert groups.size().eq(3).all() and groups.actual_up.nunique().eq(1).all()
    a=groups.agg(prediction_probability=('probability','mean'),actual_up=('actual_up','first')).reset_index();b=unique[unique.method.eq('training_frequency')][keys+['probability']].rename(columns={'probability':'frequency_probability'})
    g=a.merge(b,on=keys,validate='one_to_one');return g[FEEDBACK_COLUMNS]
def decide(visible,active,decision):
    assert set(visible.columns)==set(FEEDBACK_COLUMNS)
    assert visible.cutoff.eq(active).all() and visible.date.gt(active).all() and visible.date.le(decision).all() and visible.joint_completed.le(decision).all()
    n=len(visible);tail=visible.sort_values('date').tail(16).copy();ready=n>=16
    if ready:
        tail['block']=np.repeat(['older8','recent8'],8);y=tail.actual_up.to_numpy(float);p=tail.prediction_probability.to_numpy(float);q=tail.frequency_probability.to_numpy(float);delta=(p-y)**2-(q-y)**2;tail['excess_brier']=delta;a=float(delta[:8].mean());b=float(delta[8:].mean());refit=a>0 and b>0
    else:
        tail['block']='warmup';tail['excess_brier']=(tail.prediction_probability-tail.actual_up)**2-(tail.frequency_probability-tail.actual_up)**2;a=b=np.nan;refit=False
    return dict(available_mature_weeks=n,ready=ready,older8_excess=a,recent8_excess=b,refit=refit,reason='persistent_underperformance' if refit else 'no_persistent_underperformance' if ready else 'warmup'),tail
def visible_feedback(ledger,active,decision):
    return ledger[ledger.cutoff.eq(active)&ledger.date.gt(active)&ledger.date.le(decision)&ledger.joint_completed.le(decision)][FEEDBACK_COLUMNS].copy()
def simulate(feedback,stop_at=None):
    assert list(feedback.columns)==FEEDBACK_COLUMNS
    dates=cfg()['decision_dates'];dates=[d for d in dates if stop_at is None or d<=stop_at];ledger=[];events=[];members=[];active=None;previous=None
    for decision in dates:
        if active is not None:
            issued=feedback[feedback.cutoff.eq(active)&feedback.date.gt(previous)&feedback.date.le(decision)].sort_values('date');ledger.extend(issued.to_dict('records'))
        current=pd.DataFrame(ledger,columns=FEEDBACK_COLUMNS);visible=visible_feedback(current,active,decision) if active is not None else current
        if decision.endswith('12-31'):
            result=dict(available_mature_weeks=len(visible),ready=False,older8_excess=np.nan,recent8_excess=np.nan,refit=True,reason='annual');mandatory=True
        else:
            result,tail=decide(visible[FEEDBACK_COLUMNS],active,decision);mandatory=False
            members.extend(dict(decision_cutoff=decision,**r) for r in tail.to_dict('records'))
        selected=decision if result['refit'] else active;events.append(dict(cutoff=decision,previous_model_cutoff=active or '',selected_model_cutoff=selected,mandatory=mandatory,**result));active=selected;previous=decision
    if stop_at is None:
        issued=feedback[feedback.cutoff.eq(active)&feedback.date.gt(previous)&feedback.date.le(cfg()['label_end'])].sort_values('date');ledger.extend(issued.to_dict('records'))
    return pd.DataFrame(events),pd.DataFrame(ledger,columns=FEEDBACK_COLUMNS),pd.DataFrame(members,columns=['decision_cutoff']+FEEDBACK_COLUMNS+['block','excess_brier'])
def route_predictions(unique,ledger):
    g=unique.drop(columns='history').merge(ledger[['row_index','date','cutoff']],on=['row_index','date','cutoff'],validate='many_to_one');g.insert(0,'history','error16');assert len(g)==4352 and not g.duplicated(['method','seed','date']).any();return g
def runtime_budget(events):
    old=pd.read_csv(V28/'results/budgets.csv');old=old[old.history.eq('rolling5')];used=sorted(events.selected_model_cutoff.unique());g=old[old.cutoff.isin(used)]
    return dict(history='error16',refit_dates=len(used),neural_fits_if_run_online=3*len(used),head_fits_if_run_online=12*len(used),optimizer_steps_if_run_online=int(g.steps_per_seed.sum())*3,sample_presentations_if_run_online=int(g.total_presentations_per_seed.sum())*3,actual_new_neural_fits=0,actual_new_head_fits=0)
def next_quarter_diagnostics(feedback,events):
    out=[]
    for r in events[~events.mandatory].itertuples():
        idx=cfg()['decision_dates'].index(r.cutoff);end=cfg()['decision_dates'][idx+1] if idx+1<len(cfg()['decision_dates']) else cfg()['label_end'];current=feedback[feedback.cutoff.eq(r.previous_model_cutoff)&feedback.date.gt(r.cutoff)&feedback.date.le(end)].sort_values('date');new=feedback[feedback.cutoff.eq(r.cutoff)&feedback.date.gt(r.cutoff)&feedback.date.le(end)].sort_values('date');np.testing.assert_array_equal(current.date,new.date);assert len(current)>0
        y=current.actual_up.to_numpy();a=current.prediction_probability.to_numpy();b=new.prediction_probability.to_numpy();old_loss=(a-y)**2;new_loss=(b-y)**2;freq_loss=(current.frequency_probability.to_numpy()-y)**2
        out.append(dict(cutoff=r.cutoff,incumbent_cutoff=r.previous_model_cutoff,triggered=r.refit,ready=r.ready,reason=r.reason,next_n=len(current),incumbent_next_brier=float(old_loss.mean()),fresh_next_brier=float(new_loss.mean()),fresh_minus_incumbent_brier=float((new_loss-old_loss).mean()),incumbent_next_excess_brier=float((old_loss-freq_loss).mean()),incumbent_correct=int(((a>.5)==y).sum()),fresh_correct=int(((b>.5)==y).sum()),diagnostic_only=True))
    return pd.DataFrame(out)
