"""Prospective registry and immutable local journal. No trading integration."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
import hashlib,json,os,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; OUT=ROOT/'results'
RUNTIME=PROJECT/'prospective_r49'; TZ=timezone(timedelta(hours=8))
U='rolling5_annual20'; Q='weekly_state_validated'; W='annual_head_timeweight2y'
I='annual_head_weighted_intercept'; S='annual_head_weighted_slopes'
HISTORIES=[U,Q,W,I,S]
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset']
PRIMARY=METHODS[1:]; SEEDS=[20260910,20260911,20260912]
STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high']
def utc():return datetime.now(timezone.utc)
def iso(t):return t.astimezone(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def encoded(obj):return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def exclusive(p,raw):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def save(p,obj):exclusive(p,encoded(obj))
def cfg():return read(ROOT/'protocol.json')
def guard(ok,message):
    if not ok:raise ValueError(message)
def check_freeze():
    m=read(OUT/'freeze.json')
    guard(m['protocol_sha256']==sha(ROOT/'protocol.json'),'Protocol changed')
    for n,h in m['immutable_files'].items():guard(sha(PROJECT/n)==h,f'Frozen dependency changed: {n}')
    return m
def quarter_for(date):
    d=pd.Timestamp(date); return (d.to_period('Q').start_time-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
def annual_for(date):return f'{pd.Timestamp(date).year-1}-12-31'
def record_window(date,at):
    local=at.astimezone(TZ); d=pd.Timestamp(date)
    return d.dayofweek==4 and local.strftime('%Y-%m-%d')==date and 18<=local.hour<23

class Journal:
    def __init__(self,path=RUNTIME):self.root=Path(path)
    def initialize(self,scope='prospective'):
        guard(scope in ['prospective','synthetic'],'Unknown journal scope')
        save(self.root/'genesis.json',dict(scope=scope,created_utc=iso(utc()),protocol_sha256=sha(ROOT/'protocol.json'),source_freeze_sha256=sha(OUT/'freeze.json')))
        (self.root/'events').mkdir();(self.root/'blobs').mkdir()
    @contextmanager
    def locked(self):
        lock=self.root/'writer.lock';exclusive(lock,b'active writer\n')
        try:yield
        finally:lock.unlink()
    def events(self):
        g=read(self.root/'genesis.json')
        guard(g['protocol_sha256']==sha(ROOT/'protocol.json'),'Journal protocol mismatch')
        guard(g['source_freeze_sha256']==sha(OUT/'freeze.json'),'Journal source freeze mismatch')
        previous=sha(self.root/'genesis.json'); result=[];seen=set();last_time=None
        for seq,path in enumerate(sorted((self.root/'events').glob('*.json')),1):
            e=read(path);guard(e['sequence']==seq and path.name==f'{seq:06d}.json','Journal sequence gap')
            guard(e['previous_sha256']==previous and e['scope']==g['scope'],'Journal chain mismatch')
            at=datetime.fromisoformat(e['recorded_utc']);guard(at.tzinfo is not None,'Naive timestamp')
            guard(last_time is None or at>=last_time,'Journal time moved backwards');last_time=at
            key=(e['kind'],e['key']);guard(key not in seen,'Duplicate journal key');seen.add(key)
            for name,h in e.get('blobs',{}).items():
                guard(Path(name).name==name and sha(self.root/'blobs'/name)==h,'Journal blob changed')
            previous=sha(path);result.append(e)
        return result
    def append(self,kind,key,payload,blobs=None):
        # Production timestamps are taken here, never accepted from a CLI argument.
        with self.locked():
            events=self.events();guard(not any(e['kind']==kind and e['key']==key for e in events),'Duplicate journal key')
            g=read(self.root/'genesis.json');at=utc()
            guard(not events or at>=datetime.fromisoformat(events[-1]['recorded_utc']),'System clock moved backwards')
            if kind=='prediction':
                guard(g['scope']=='prospective','Synthetic journal cannot receive prospective prediction')
                guard(record_window(key,at),'Prediction write is outside the Friday 18:00-23:00 CST window')
                guard(key in cfg()['signal_slots'],'Signal is outside frozen cohort')
            if kind=='label':guard(any(e['kind']=='prediction' and e['key']==key for e in events),'Label has no prior prediction')
            e=dict(sequence=len(events)+1,previous_sha256=sha(self.root/'events'/f'{len(events):06d}.json') if events else sha(self.root/'genesis.json'),recorded_utc=iso(at),scope=g['scope'],kind=kind,key=key,payload=payload,blobs=blobs or {})
            save(self.root/'events'/f'{e["sequence"]:06d}.json',e)
        return e
    def blob(self,raw,suffix):
        name=digest(raw)+suffix;p=self.root/'blobs'/name
        if p.exists():guard(p.read_bytes()==raw,'Blob digest collision')
        else:exclusive(p,raw)
        return name,digest(raw)

def old_evidence():
    previous=PROJECT/'research_v48';m=read(previous/'results/preparation_manifest.json');old=dict(m['old_evidence'])
    old.update({str((previous/n).relative_to(PROJECT)):h for n,h in read(previous/'results/delivery_manifest.json')['files'].items()})
    p=previous/'results/delivery_manifest.json';old[str(p.relative_to(PROJECT))]=sha(p)
    guard(len(old)==6206,'Unexpected old evidence count')
    for n,h in old.items():guard(sha(PROJECT/n)==h,f'Old evidence changed: {n}')
    return old
