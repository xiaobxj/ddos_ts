"""Read-only audit of archived state correction validation evidence."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V40=PROJECT/'research_v40';V39=PROJECT/'research_v39'
sys.path.insert(0,str(V40));import common40 as previous_round
prior=previous_round.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data
STATES=previous_round.STATES;LEARNED=previous_round.LEARNED;PRIMARY=previous_round.PRIMARY;ANNUAL=previous_round.ANNUAL;CADENCES=['monthly','quarterly']
HISTORY={'monthly':('monthly_state_validated','monthly_state_ungated'),'quarterly':('weekly_state_validated','weekly_state_ungated')}
COPIES=[(V40/'results'/f'{n}.csv',f'{n}.csv') for n in ['model_predictions','ensemble_predictions','ensemble_metrics','seed_metrics','state_metrics','weekly_signal_bank']]
for source,target in [('gate_decisions','gates'),('schedule','schedule'),('split_membership','members'),('validation_predictions','validation_predictions'),('month_outcomes','outcomes')]:COPIES.append((V40/'results'/f'{source}.csv',f'monthly_{target}.csv'))
COPIES.append((V40/'results/correction_heads.json','monthly_heads.json'))
for source,target in [('quarterly_gates','gates'),('quarterly_schedule','schedule'),('quarterly_members','members')]:COPIES.append((V40/'results'/f'{source}.csv',f'quarterly_{target}.csv'))
COPIES.extend([(V40/'results/quarterly_heads.json','quarterly_heads.json'),(V39/'results/validation_predictions.csv','quarterly_validation_predictions.csv'),(V39/'results/quarter_outcomes.csv','quarterly_outcomes.csv')])
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    x=previous_round.source_hashes();x.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')});return x
def input_hashes():
    x=previous_round.input_hashes();paths=[a for a,b in COPIES]+[V40/'protocol.json']+[V40/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']];x.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return x
def old_evidence():
    x=dict(read(V40/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V40/'results/delivery_manifest.json')['files'].items():x[str((V40/n).relative_to(PROJECT))]=d
    p=V40/'results/delivery_manifest.json';x[str(p.relative_to(PROJECT))]=sha(p);assert len(x)==5633
    for n,d in x.items():assert sha(PROJECT/n)==d,n
    return x
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

def cell_gate(mode,train_n,n,brier_difference,correct_difference):
    reason=mode if mode!='ready' else 'insufficient_state_train' if train_n<10 else 'insufficient_validation' if n<5 else 'brier_not_better' if not brier_difference < -1e-12 else 'direction_worse' if correct_difference<0 else 'accepted'
    return reason=='accepted',reason
def eq(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<tol,(a,b)
