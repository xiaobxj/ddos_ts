"""Mature weekly calibration with purged chronological validation and exact fallback."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V38=PROJECT/'research_v38'
sys.path.insert(0,str(V38));import common38 as previous_round
v37=previous_round.previous_round;prior=previous_round.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data;probability=prior.probability;metric=prior.metric;statistics=prior.statistics
STATES=previous_round.STATES;LEARNED=previous_round.LEARNED;PRIMARY=previous_round.PRIMARY;ANNUAL=previous_round.ANNUAL
POLICIES={'state_gated':'weekly_state_validated','state_direct':'weekly_state_ungated','global_gated':'weekly_global_validated','global_direct':'weekly_global_ungated'};NEW=list(POLICIES.values());FAMILIES=['state','global']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V38/'protocol.json']+[V38/'results'/n for n in ['delivery_manifest.json','preparation_manifest.json','verification.json','结果解读与下一步.md','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','routing.csv','weekly_context.csv','seed_correction_effects.csv']];r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V38/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V38/'results/delivery_manifest.json')['files'].items():r[str((V38/n).relative_to(PROJECT))]=d
    p=V38/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5484
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==p['source_sha256'] and c['protocol_sha256']==p['protocol_sha256']
        for n,d in c['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def periods():return cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end=cfg()['label_end'],n=272)]+[dict(name=f'year_{y}',start=f'{y}-01-01',end=f'{y}-12-31') for y in range(2021,2027)]
def split(pool,cutoff):
    settings=cfg()['settings'];past=pool[pool.joint_completed.le(cutoff)].sort_values('date');val=past.tail(settings['validation_weeks']);fit_cutoff=(pd.Timestamp(val.date.iloc[0])-pd.Timedelta(days=1)).strftime('%Y-%m-%d') if len(val) else None
    train=pool[pool.joint_completed.le(fit_cutoff)].sort_values('date').tail(settings['training_weeks']) if fit_cutoff else pool.iloc[:0]
    ready=len(train)==settings['training_weeks'] and len(val)==settings['validation_weeks'];mode='annual_reset' if cutoff.endswith('12-31') else 'ready' if ready else 'insufficient_history'
    return train,val,dict(cutoff=cutoff,fit_cutoff=fit_cutoff,mode=mode,available=len(past),train_n=len(train),validation_n=len(val),train_max_maturity=train.joint_completed.max() if len(train) else None,validation_first=val.date.min() if len(val) else None,validation_last=val.date.max() if len(val) else None,validation_max_maturity=val.joint_completed.max() if len(val) else None,purged_between_n=int((past.date.lt(val.date.min())&past.joint_completed.gt(fit_cutoff)).sum()) if fit_cutoff else 0)
def input_rows(bank,ids,method,seed):
    r=bank[bank.method.eq(method)&bank.seed.eq(seed)].set_index('row_index').loc[list(ids)].reset_index();assert r.row_index.tolist()==list(ids);return r
def fitted(train,mode):
    s=cfg()['settings'];rows=[]
    for family,components in [('state',STATES),('global',['all'])]:
        for component in components:
            g=train if component=='all' else train[train.state.eq(component)];eligible=mode=='ready' and (len(g)>=s['minimum_state_train'] if family=='state' else len(g)==s['training_weeks'])
            r=v37.solve(g.annual_logit.to_numpy(),g.actual_up.to_numpy(float),s['ridge_sum_lambda'],s['absolute_logit_cap'],s['bisection_iterations']) if eligible else dict(offset=0.,status=mode if mode!='ready' else 'insufficient_state_train',n=len(g),up_n=int(g.actual_up.sum()),lambda_sum=s['ridge_sum_lambda'])
            rows.append(dict(family=family,component=component,fit_eligible=eligible,**r))
    return rows
def gate_decision(mode,family,train_n,validation_n,p0,p,y):
    s=cfg()['settings'];n=len(y);assert n==validation_n;bc=int(((p0>.5)==y).sum());cc=int(((p>.5)==y).sum());delta=float(np.mean((p-y)**2-(p0-y)**2)) if n else None
    reason=mode if mode!='ready' else 'insufficient_state_train' if family=='state' and train_n<s['minimum_state_train'] else 'insufficient_validation' if validation_n<(s['minimum_state_validation'] if family=='state' else s['validation_weeks']) else 'brier_not_better' if not delta < -s['brier_improvement_tolerance'] else 'direction_worse' if cc<bc else 'accepted'
    return dict(accepted=reason=='accepted',reason=reason,validation_n=n,baseline_correct=bc,candidate_correct=cc,brier_difference=delta,baseline_brier=float(((p0-y)**2).mean()) if n else None,candidate_brier=float(((p-y)**2).mean()) if n else None)
def corrected_probability(z,p0,delta):return np.where(np.asarray(delta)==0,p0,probability(np.asarray(z)+delta))
