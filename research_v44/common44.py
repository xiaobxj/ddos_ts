"""Frozen information-arrival feasibility for each original accepted source."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; OUT=ROOT/'results'; PREV=PROJECT/'research_v43'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset']
PRIMARY=METHODS[1:]; STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high']
VIEWS=['cumulative','rolling13']
COPIES=['weekly_signal_bank.csv','split_membership.csv','schedule.csv','gate_decisions.csv','correction_heads.json','validation_predictions.csv','retention_decisions.csv','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cfg(): return read(ROOT/'protocol.json')
def now(): return pd.Timestamp.now(tz='UTC').isoformat()
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(n): return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=dict(read(PREV/'results/preparation_manifest.json')['source_sha256'])
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')}); return r
def old_evidence():
    r=dict(read(PREV/'results/preparation_manifest.json')['old_evidence'])
    r.update({str((PREV/n).relative_to(PROJECT)):h for n,h in read(PREV/'results/delivery_manifest.json')['files'].items()})
    p=PREV/'results/delivery_manifest.json'; r[str(p.relative_to(PROJECT))]=sha(p)
    assert len(r)==5800
    for n,h in r.items(): assert sha(PROJECT/n)==h,n
    return r
def inputs():
    paths=[PREV/'results'/n for n in COPIES+['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']]+[PREV/'protocol.json']
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
def manifest(phase): return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=inputs(),executable=sys.executable,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra); save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json'); assert p['source_sha256']==source_hashes() and p['protocol_sha256']==sha(ROOT/'protocol.json') and p['input_sha256']==inputs()
    for n,h in p['artifacts'].items(): assert sha(ROOT/n)==h,n
    if contract:
        c=read(OUT/'contract_verification.json'); assert c['status']=='PASS'
        assert c['source_sha256']==p['source_sha256'] and c['protocol_sha256']==p['protocol_sha256']
        for n,h in c['artifacts'].items(): assert sha(ROOT/n)==h,n
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json'); assert r['finished_utc']
    for n,h in r['artifacts'].items(): assert sha(ROOT/n)==h,n
    return r
def period_definitions():
    return [('all_2020_2026','2020-01-01','2026-12-31'),('early_2021_2023','2021-01-01','2023-12-31'),('recent_2024_2026','2024-01-01','2026-08-31')]+[(f'year_{y}',f'{y}-01-01',f'{y}-12-31') for y in range(2021,2027)]

META=['row_index','date','joint_completed','state','year','encoder_cutoff']
CLASSES=['reached_before_expiry','reached_only_at_expiry','not_reached_by_expiry','observation_censored']
def expiry_after(source):return (pd.Timestamp(source)+pd.offsets.QuarterEnd(1)).strftime('%Y-%m-%d')
def checkpoints(source,expiry):return [x.strftime('%Y-%m-%d') for x in pd.date_range(pd.Timestamp(source)+pd.offsets.MonthEnd(1),expiry,freq='ME')]
def canonical_metadata(bank):
    for col in META[1:]:assert bank.groupby('row_index')[col].nunique().eq(1).all(),col
    r=bank[META].drop_duplicates().sort_values(['joint_completed','date','row_index']).reset_index(drop=True)
    assert not r.row_index.duplicated().any();return r
def count_members(meta,source,state,cutoff):
    assert not meta.row_index.duplicated().any()
    mature=meta[meta.joint_completed.le(cutoff)]
    recent=set(mature.sort_values(['date','row_index']).tail(13).row_index)
    selected=mature[mature.joint_completed.gt(source)&mature.state.eq(state)&mature.year.eq(int(source[:4]))].copy()
    selected['in_rolling13']=selected.row_index.isin(recent)
    return selected.sort_values(['joint_completed','date','row_index'])
def classify_track(rows,expiry,end,view):
    key='cumulative_n' if view=='cumulative' else 'rolling13_n'
    hits=[r for r in rows if r['observed'] and r[key]>=5]
    before=[r for r in hits if r['checkpoint']<expiry]
    if before:return 'reached_before_expiry',before[0]['checkpoint'],hits[0]['checkpoint']
    if expiry>end:return 'observation_censored',None,hits[0]['checkpoint'] if hits else None
    return ('reached_only_at_expiry',None,hits[0]['checkpoint']) if hits else ('not_reached_by_expiry',None,None)
