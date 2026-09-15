"""Frozen, read-only revalidation of the original R42 incoming saved record."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; OUT=ROOT/'results'; PREV=PROJECT/'research_v42'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset']
PRIMARY=METHODS[1:]; STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high']
VIEWS=['full_validation','post_source_validation','incremental_validation']
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
    assert len(r)==5754
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
def evidence(n,bd,cd,available=True):
    if not available:return 'unfitted'
    if n==0:return 'no_samples'
    if n<5:return 'insufficient_validation'
    if not bd < -1e-12:return 'brier_not_better'
    if cd<0:return 'direction_worse'
    return 'pass'
def comparison(old,current):
    reject=['brier_not_better','direction_worse']
    if old in ['no_saved_record','annual_mismatch']:return 'not_assessed'
    if current=='unfitted':return 'current_unfitted'
    if old in ['no_samples','insufficient_validation']:return 'insufficient_evidence'
    return 'both_pass' if old==current=='pass' else 'old_only_pass' if old=='pass' and current in reject else 'current_only_pass' if current=='pass' and old in reject else 'both_reject'
def summarize(cells,weeks):
    rows=[]
    for period,start,end in period_definitions():
        for (method,availability,view),g in cells[cells.cutoff.between(start,end)&cells.availability.isin(['in_life','expired'])].groupby(['method','availability','view']):
            ids=set(g.subject_id); w=weeks[weeks.subject_id.isin(ids)]
            if view=='post_source_validation':w=w[w.matured_after_source]
            if view=='incremental_validation':w=w[w.matured_since_previous_decision]
            unique=w.row_index.nunique(); source_unique=len(w[['source_cutoff','row_index']].drop_duplicates())
            row=dict(period=period,method=method,availability=availability,view=view,cells=len(g),validation_uses=int(g.n.sum()),unique_calendar_weeks=unique,unique_source_week_pairs=source_unique,repeated_source_week_uses=len(w)-source_unique)
            for label in ['pass','brier_not_better','direction_worse','no_samples','insufficient_validation']:row['old_'+label]=int(g.old_evidence.eq(label).sum())
            for label in ['both_pass','old_only_pass','current_only_pass','both_reject','insufficient_evidence','current_unfitted']:row[label]=int(g.comparison.eq(label).sum())
            quality=g[g.original_reason.isin(['brier_not_better','direction_worse'])]
            row.update(quality_clear_cells=len(quality),quality_clear_old_pass=int(quality.old_evidence.eq('pass').sum()),quality_clear_old_reject=int(quality.old_evidence.isin(['brier_not_better','direction_worse']).sum()),quality_clear_insufficient=int(quality.old_evidence.isin(['no_samples','insufficient_validation']).sum()))
            rows.append(row)
    return pd.DataFrame(rows)
