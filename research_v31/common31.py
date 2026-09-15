"""Quarter-long paired shadow trials; immutable annual and quarterly models."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V30=PROJECT/'research_v30';V28=PROJECT/'research_v28'
sys.path.insert(0,str(V30));import common30 as previous_round
prior=previous_round.prior
np=prior.np;pd=prior.pd;torch=prior.torch;training=prior.training
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;data=prior.data;statistics=prior.statistics
METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
training_rows=previous_round.training_rows;ensemble_from=previous_round.ensemble_from
HISTORIES=previous_round.HISTORIES+['shadow_trial','annual_quarter_half'];NEW=HISTORIES[-2:]
PAIR_COLUMNS=['row_index','date','joint_completed','trial_start','incumbent_cutoff','challenger_cutoff','incumbent_probability','challenger_probability','actual_up']

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V30/'protocol.json']+[V30/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','universal_model_predictions.csv','model_predictions.csv','ensemble_predictions.csv','budgets.csv']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V30/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V30/'results/delivery_manifest.json')['files'].items():r[str((V30/n).relative_to(PROJECT))]=d
    p=V30/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4696
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def archive_predictions():return pd.read_csv(V30/'results/universal_model_predictions.csv',float_precision='round_trip')
def baselines():return tuple(pd.read_csv(V30/'results'/n,float_precision='round_trip') for n in ['model_predictions.csv','ensemble_predictions.csv'])
def feedback_from(unique):return previous_round.feedback_from(unique)
def visible_pairs(ledger,start,incumbent,challenger,decision):
    return ledger[ledger.trial_start.eq(start)&ledger.incumbent_cutoff.eq(incumbent)&ledger.challenger_cutoff.eq(challenger)&ledger.date.gt(start)&ledger.date.le(decision)&ledger.joint_completed.le(decision)][PAIR_COLUMNS].sort_values('date').copy()
def decide(visible,start,incumbent,challenger,decision):
    assert list(visible.columns)==PAIR_COLUMNS and visible.date.is_unique
    assert visible.trial_start.eq(start).all() and visible.incumbent_cutoff.eq(incumbent).all() and visible.challenger_cutoff.eq(challenger).all()
    assert visible.date.gt(start).all() and visible.date.le(decision).all() and visible.joint_completed.le(decision).all()
    assert visible.incumbent_cutoff.lt(visible.date).all() and visible.challenger_cutoff.lt(visible.date).all()
    g=visible.copy();y=g.actual_up.to_numpy(float);a=g.incumbent_probability.to_numpy(float);b=g.challenger_probability.to_numpy(float)
    g['challenger_minus_incumbent_brier']=(b-y)**2-(a-y)**2;g['challenger_minus_incumbent_error']=((b>.5)!=y).astype(int)-((a>.5)!=y).astype(int)
    n=len(g);ready=incumbent!=challenger and n>=cfg()['shadow']['min_paired_weeks'];delta=float(g.challenger_minus_incumbent_brier.mean()) if n else np.nan;gain=int(-g.challenger_minus_incumbent_error.sum());promote=bool(ready and delta<0 and gain>0)
    reason='same_model' if incumbent==challenger else 'warmup' if not ready else 'promote' if promote else 'hold'
    return dict(paired_mature_weeks=n,ready=ready,challenger_minus_incumbent_brier=delta,challenger_correct_gain=gain,promote=promote,reason=reason),g
def issued_pairs(feedback,start,end,incumbent,challenger):
    a=feedback[feedback.cutoff.eq(incumbent)&feedback.date.gt(start)&feedback.date.le(end)].sort_values('date');b=feedback[feedback.cutoff.eq(challenger)&feedback.date.gt(start)&feedback.date.le(end)].sort_values('date')
    assert a.row_index.tolist()==b.row_index.tolist() and a.actual_up.tolist()==b.actual_up.tolist()
    g=a[['row_index','date','joint_completed','actual_up']].copy();g['trial_start']=start;g['incumbent_cutoff']=incumbent;g['challenger_cutoff']=challenger;g['incumbent_probability']=a.prediction_probability.to_numpy();g['challenger_probability']=b.prediction_probability.to_numpy();return g[PAIR_COLUMNS]
def simulate(feedback,stop_at=None):
    decisions=[d for d in cfg()['decision_dates'] if stop_at is None or d<=stop_at];ledger=pd.DataFrame(columns=PAIR_COLUMNS);events=[];members=[];incumbent=challenger=start=None
    for decision in decisions:
        if start is not None:
            new=issued_pairs(feedback,start,decision,incumbent,challenger);ledger=new.reset_index(drop=True) if ledger.empty else pd.concat([ledger,new],ignore_index=True)
        visible=visible_pairs(ledger,start,incumbent,challenger,decision)
        mandatory=decision.endswith('12-31')
        if mandatory:r=dict(paired_mature_weeks=len(visible),ready=False,challenger_minus_incumbent_brier=np.nan,challenger_correct_gain=0,promote=False,reason='annual')
        else:
            r,g=decide(visible,start,incumbent,challenger,decision);members.extend(dict(decision_cutoff=decision,**row) for row in g.to_dict('records'))
        selected=decision if mandatory else challenger if r['promote'] else incumbent
        events.append(dict(cutoff=decision,trial_start=start or '',incumbent_cutoff=incumbent or '',evaluated_challenger_cutoff=challenger or '',selected_model_cutoff=selected,next_challenger_cutoff=decision,mandatory=mandatory,**r));incumbent=selected;challenger=decision;start=decision
    if stop_at is None:ledger=pd.concat([ledger,issued_pairs(feedback,start,cfg()['label_end'],incumbent,challenger)],ignore_index=True)
    return pd.DataFrame(events),ledger,pd.DataFrame(members,columns=['decision_cutoff']+PAIR_COLUMNS+['challenger_minus_incumbent_brier','challenger_minus_incumbent_error'])
def shadow_route(ledger):return ledger[['row_index','date','incumbent_cutoff']].rename(columns={'incumbent_cutoff':'cutoff'})
def route_predictions(unique,route):
    r=unique.drop(columns='history').merge(route,on=['row_index','date','cutoff'],validate='many_to_one');r.insert(0,'history','shadow_trial');assert len(r)==4352 and not r.duplicated(['method','seed','date']).any();return r
def half_blend(base):
    keys=['row_index','date','method','seed'];a=base[base.history.eq('rolling5_annual20')].sort_values(keys).reset_index(drop=True);b=base[base.history.eq('rolling5_quarterly20')].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys+['actual','actual_up','joint_completed']],b[keys+['actual','actual_up','joint_completed']],check_exact=True)
    r=a.copy();r['history']='annual_quarter_half';r['cutoff']=b.cutoff;r['score']=(a.score+b.score)*.5;r['probability']=(a.probability+b.probability)*.5;r['direction_up']=np.where(r.method.eq('native_mse'),r.score>0,r.probability>.5).astype(int)
    route=a[['row_index','date','cutoff']].rename(columns={'cutoff':'annual_cutoff'});route['quarterly_cutoff']=b.cutoff;route['annual_weight']=.5;route['quarterly_weight']=.5;route=route.drop_duplicates().reset_index(drop=True);assert len(route)==272 and len(r)==4352
    return r,route
def runtime_budgets(events):
    b=pd.read_csv(V30/'results/budgets.csv');q=b[b.history.eq('rolling5_quarterly20')].iloc[0].to_dict();rows=[]
    for h in NEW:
        r=dict(q,history=h);r['selected_incumbent_dates']=int(events.selected_model_cutoff.nunique()) if h=='shadow_trial' else np.nan;rows.append(r)
    return pd.concat([b,pd.DataFrame(rows)],ignore_index=True)
def next_quarter_diagnostics(feedback,events):
    rows=[]
    for e in events[(~events.mandatory)&events.incumbent_cutoff.ne(events.evaluated_challenger_cutoff)].itertuples():
        i=cfg()['decision_dates'].index(e.cutoff);end=cfg()['decision_dates'][i+1] if i+1<len(cfg()['decision_dates']) else cfg()['label_end'];a=feedback[feedback.cutoff.eq(e.incumbent_cutoff)&feedback.date.gt(e.cutoff)&feedback.date.le(end)].sort_values('date');b=feedback[feedback.cutoff.eq(e.evaluated_challenger_cutoff)&feedback.date.gt(e.cutoff)&feedback.date.le(end)].sort_values('date');assert a.date.tolist()==b.date.tolist()
        y=a.actual_up.to_numpy();p=a.prediction_probability.to_numpy();q=b.prediction_probability.to_numpy();rows.append(dict(cutoff=e.cutoff,incumbent_cutoff=e.incumbent_cutoff,challenger_cutoff=e.evaluated_challenger_cutoff,promoted=e.promote,next_n=len(a),incumbent_correct=int(((p>.5)==y).sum()),challenger_correct=int(((q>.5)==y).sum()),challenger_minus_incumbent_brier=float(((q-y)**2-(p-y)**2).mean()),diagnostic_only=True))
    return pd.DataFrame(rows)
