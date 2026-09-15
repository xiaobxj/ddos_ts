"""Offline safety and maturity tests, with all simulated records isolated in temp dirs."""
from experiment3 import *
import daily3 as d
import tempfile,copy
from unittest.mock import patch

def rejected(fn):
    try:fn()
    except (AssertionError,RuntimeError):return
    raise AssertionError('Expected rejection did not occur')

def seed(root,date,h=5):
    issued=datetime.datetime.fromisoformat(date+'T12:00:00+00:00');events=[]
    d.append(root,events,'genesis',dict(simulation=True),issued)
    d.append(root,events,'forecast',dict(signal_date=date,rows=[dict(name='h5.quarterly.raw',h=h,probability=.6)],targets={str(h):d.target_date(date,h)}),issued)
    return events

def main():
    f=data();checks=[]
    with tempfile.TemporaryDirectory(prefix='daily3_test_',dir=OUT) as temporary:
        root=Path(temporary).resolve();assert root.is_relative_to(OUT.resolve()) and root.name.startswith('daily3_test_')
        ledger=root/'ledger';events=seed(ledger,'2026-09-08')
        assert d.chain(ledger)==events
        with d.locked(root/'locks'):rejected(lambda:d.run(root/'locks'))
        checks.append('exclusive concurrent-run lock')
        before=f[f.date.le('2026-09-14')].reset_index(drop=True);assert d.settle(events,before)==[]
        labels=d.settle(events,f);assert len(labels)==1 and labels[0]['target_date']=='2026-09-15'
        expected=f.loc[f.date.eq('2026-09-15'),'raw_close'].iloc[0]/f.loc[f.date.eq('2026-09-08'),'raw_close'].iloc[0]-1
        assert abs(labels[0]['actual_return']-expected)<1e-12
        d.append(ledger,events,'label',labels[0],datetime.datetime.fromisoformat('2026-09-15T12:00:00+00:00'))
        assert d.settle(events,f)==[];checks.extend(['no immature labels','independent H5 close-to-close arithmetic','idempotent settlement'])
        missing=f.copy();missing.loc[missing.date.eq('2026-09-11'),'valid_ohlc']=False
        assert d.settle(events[:2],missing)[0]['status']=='UNSCORABLE_MISSING_BAR';checks.append('halt/missing bar never compresses calendar or becomes a win')
        dividend=seed(root/'dividend','2026-05-25',1);lab=d.settle(dividend,f)[0]
        expected=(f.loc[f.date.eq('2026-05-26'),'raw_close'].iloc[0]+.75)/f.loc[f.date.eq('2026-05-25'),'raw_close'].iloc[0]-1
        assert abs(lab['actual_return']-expected)<1e-12;checks.append('cash dividend entitlement across ex-date')
        assert d.target_date('2026-09-24',1)=='2026-09-28' and d.target_date('2026-09-24',5)=='2026-10-09';checks.append('official holiday-aware target dates')
        quote=['']*35;quote[30]='20260915161451'
        for col,k in [('raw_open',5),('raw_close',3),('raw_high',33),('raw_low',34)]:quote[k]=str(f[col].iloc[-1])
        pkg=dict(valid_until='2026-09-30');loc=datetime.datetime.fromisoformat('2026-09-15T20:00:00+08:00')
        assert d.issue_reason(loc,f,quote,pkg,[])==''
        assert d.issue_reason(loc.replace(hour=14),f,quote,pkg,[])
        assert d.issue_reason(loc.replace(day=16),f,quote,pkg,[])
        assert d.issue_reason(loc.replace(month=10,day=8),f,quote,pkg,[])
        today=seed(root/'today','2026-09-15');assert d.issue_reason(loc,f,quote,pkg,today)
        q=quote.copy();q[30]='20260914161451';assert d.issue_reason(loc,f,q,pkg,[])
        checks.extend(['pre-close blocked','stale-day backfill blocked','package expiry blocked','same-day duplicate blocked','stale quote blocked'])
        changed=load_csv(V2/'data/raw.csv');base=changed.copy();changed.loc[changed.date.eq('2026-09-10'),'close']+=.01
        rejected(lambda:d.merge_unchanged(base,changed[changed.date.ge('2026-01-01')],'2026-09-15',['open','high','low','close','volume']))
        checks.append('historical vendor revision blocked')
        path=ledger/'events/00000002.json';original=path.read_bytes();obj=read(path);obj['payload']['rows'][0]['probability']=.9;path.write_bytes(d.canonical(obj))
        rejected(lambda:d.chain(ledger));path.write_bytes(original);checks.append('event tamper detected')
        key=d.blob(ledger,b'original');(ledger/'blobs'/key).write_bytes(b'changed');rejected(lambda:d.getblob(ledger,key));checks.append('snapshot tamper detected')
        expiryroot=root/'expiry';seed(expiryroot,'2026-09-08')
        clock=datetime.datetime.fromisoformat('2026-10-08T12:00:00+00:00')
        with patch.object(d,'package_check',return_value=pkg),patch.object(d,'utc_clock',return_value=clock),patch.object(d,'download',return_value=dict(quote=quote)),patch.object(d,'frames',return_value=(f,dict(simulation=True))),patch.object(d,'infer',side_effect=AssertionError('Expired model used')):
            expired=d.run(expiryroot)
        assert sum(e['kind']=='forecast' for e in expired)==1 and sum(e['kind']=='label' for e in expired)==1
        checks.append('expired package still settles existing records without inference')
    path=OUT/('daily_unit_verified_final.json' if (OUT/'daily_unit_verified.json').exists() else 'daily_unit_verified.json')
    save(path,dict(status='PASS',completed_utc=now(),source_sha256=sha(ROOT/'daily3.py'),checks=checks,simulation='temporary test directories only; no production events created'))

if __name__=='__main__':main()
