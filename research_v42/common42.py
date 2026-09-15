"""Bounded historical retention with immutable source candidates and gates."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V41=PROJECT/'research_v41';V40=PROJECT/'research_v40'
sys.path.insert(0,str(V41));import common41 as previous_round
v40=previous_round.previous_round;v37=v40.v37;prior=v40.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data;metric=prior.metric;statistics=prior.statistics;probability=prior.probability
STATES=v40.STATES;LEARNED=v40.LEARNED;PRIMARY=v40.PRIMARY;ANNUAL=v40.ANNUAL;MONTHLY='monthly_state_validated';QUARTER='weekly_state_validated';NEW=['monthly_state_retained'];REFS=[MONTHLY,QUARTER,ANNUAL];REPORT_HIST=[ANNUAL,QUARTER,MONTHLY]+NEW
COPIES=[(V41/'results'/f'{s}.csv',f'{d}.csv') for s,d in [('model_predictions','baseline_model_predictions'),('ensemble_predictions','baseline_ensemble_predictions'),('ensemble_metrics','baseline_metrics'),('weekly_signal_bank','weekly_signal_bank'),('monthly_gates','gate_decisions'),('monthly_schedule','schedule'),('monthly_members','split_membership'),('monthly_validation_predictions','validation_predictions')]]
COPIES.append((V41/'results/monthly_heads.json','correction_heads.json'))
COPIES.extend([(V40/'results'/f'{s}.csv',f'{d}.csv') for s,d in [('routing','routing'),('weekly_context','weekly_context'),('seed_routing','original_monthly_seed_routing'),('weekly_policy_effects','original_monthly_weekly_effects')]])
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[a for a,b in COPIES]+[V41/'protocol.json']+[V41/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']];r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V41/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V41/'results/delivery_manifest.json')['files'].items():r[str((V41/n).relative_to(PROJECT))]=d
    p=V41/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5691
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


def expiry_after(cutoff):return (pd.Timestamp(cutoff)+pd.offsets.QuarterEnd(1)).strftime('%Y-%m-%d')
def annual_for(cutoff):return f'{int(cutoff[:4])-1}-12-31'
def eq(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<tol,(a,b)
def usable(source_cutoff,expiry,signal_date):return isinstance(source_cutoff,str) and source_cutoff<signal_date<=expiry and source_cutoff[:4]==signal_date[:4]
def retention_decisions(gates,dates):
    lookup=gates[gates.family.eq('state')].set_index(['cutoff','method','component']);records=[]
    for method in LEARNED:
        for state in STATES:
            stored=None
            for cutoff in dates:
                g=lookup.loc[(cutoff,method,state)];old=dict(stored) if stored else None;expired=bool(stored and cutoff>=stored['expiry'])
                if expired:stored=None
                if g.accepted:
                    assert g['mode']=='ready' and g.training_n>=10 and g.validation_n>=5
                    stored=dict(source_cutoff=cutoff,expiry=expiry_after(cutoff),source_encoder_cutoff=annual_for(cutoff));action='fresh';reason='accepted_current'
                elif g['mode']=='ready' and g.reason=='insufficient_validation' and g.training_n>=10 and g.validation_n<5:
                    if stored and stored['source_encoder_cutoff']==annual_for(cutoff):action='carry';reason='retained_validation_shortage'
                    else:stored=None;action='fallback';reason='retention_expired' if expired else 'no_valid_saved_acceptance'
                else:stored=None;action='fallback';reason='cleared_'+g.reason
                records.append(dict(cutoff=cutoff,method=method,state=state,mode=g['mode'],original_accepted=bool(g.accepted),original_reason=g.reason,training_n=int(g.training_n),validation_n=int(g.validation_n),previous_source_cutoff=old['source_cutoff'] if old else None,previous_expiry=old['expiry'] if old else None,previous_expired=expired,action=action,retention_reason=reason,source_cutoff=stored['source_cutoff'] if stored else None,expiry=stored['expiry'] if stored else None,source_encoder_cutoff=stored['source_encoder_cutoff'] if stored else None))
    return pd.DataFrame(records)
