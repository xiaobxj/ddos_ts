"""Manual prospective CLI. Production has no historical timestamp override."""
import argparse
from common49 import *
from data49 import snapshot,fetch,load_snapshot,label_for,unchanged_prefix
from forecast49 import Engine,package_check
from evaluate49 import summarize

def status(journal):
    check_freeze();events=journal.events();at=utc();local=at.astimezone(TZ);p=read(OUT/'bootstrap_package.json')
    future=[d for d in cfg()['signal_slots'] if d>=local.strftime('%Y-%m-%d')]
    return dict(status='WAIT_FOR_SIGNAL_WINDOW' if not record_window(local.strftime('%Y-%m-%d'),at) else 'SIGNAL_WINDOW_OPEN',now_utc=iso(at),next_scheduled_friday=future[0] if future else None,latest_archived_snapshot=max([e['payload']['end'] for e in events if e['kind']=='snapshot'],default=None),predictions=sum(e['kind']=='prediction' for e in events),labels=sum(e['kind']=='label' for e in events),bootstrap_valid_until=p['valid_until'],next_parameter_cutoff='2026-09-30',scheduler_installed=False,journal_events=len(events),journal_tip_sha256=sha(journal.root/'events'/f'{len(events):06d}.json') if events else sha(journal.root/'genesis.json'))
def record(journal):
    check_freeze();at=utc();date=at.astimezone(TZ).strftime('%Y-%m-%d')
    guard(date in cfg()['signal_slots'] and record_window(date,at),'No current Friday 18:00-23:00 CST prospective slot; backfill forbidden')
    events=journal.events();guard(not any(e['kind']=='prediction' and e['key']==date for e in events),'Prediction already committed')
    candidates=[e for e in events if e['kind']=='snapshot' and e['payload']['end']==date and datetime.fromisoformat(e['recorded_utc']).astimezone(TZ).strftime('%Y-%m-%d')==date]
    guard(candidates,'No same-day archived final-bar snapshot; run fetch or ingest first')
    source=candidates[-1];frame=load_snapshot(journal,source);p=read(OUT/'bootstrap_package.json');package_check(p,date,at)
    result=Engine(p).predict(frame,date);guard(record_window(date,utc()),'Inference crossed the recording deadline')
    payload=dict(snapshot_sequence=source['sequence'],snapshot_sha256=sha(journal.root/'events'/f'{source["sequence"]:06d}.json'),package_file=str((OUT/'bootstrap_package.json').relative_to(PROJECT)),package_sha256=sha(OUT/'bootstrap_package.json'),protocol_sha256=sha(ROOT/'protocol.json'),forecast=result)
    return journal.append('prediction',date,payload)
def settle(journal):
    check_freeze();events=journal.events();snapshots=[e for e in events if e['kind']=='snapshot'];guard(snapshots,'No snapshot for labels')
    latest=snapshots[-1];f=load_snapshot(journal,latest);done={e['key'] for e in events if e['kind']=='label'};results=[]
    for e in events:
        if e['kind']!='prediction' or e['key'] in done:continue
        original=events[e['payload']['snapshot_sequence']-1];prefix=load_snapshot(journal,original)
        unchanged_prefix(prefix,f,e['key'])
        result=label_for(f,e['key'])
        if result['status']=='PENDING':results.append(dict(date=e['key'],**result));continue
        guard(datetime.fromisoformat(e['recorded_utc']).astimezone(TZ).strftime('%Y-%m-%d')<result['entry_date'],'Prediction not before entry')
        guard(result['joint_completed']<=datetime.fromisoformat(latest['recorded_utc']).astimezone(TZ).strftime('%Y-%m-%d'),'Label snapshot predates maturity')
        result.update(snapshot_sequence=latest['sequence'],snapshot_sha256=sha(journal.root/'events'/f'{latest["sequence"]:06d}.json'),prediction_sha256=sha(journal.root/'events'/f'{e["sequence"]:06d}.json'))
        results.append(journal.append('label',e['key'],result))
    return results
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['status','fetch','ingest','record','settle','evaluate']);p.add_argument('--csv',type=Path)
    args=p.parse_args();journal=Journal()
    if args.command=='status':r=status(journal)
    elif args.command=='fetch':r=fetch(journal)
    elif args.command=='ingest':
        guard(args.csv is not None,'ingest requires --csv');raw=args.csv.read_bytes();r=snapshot(journal,pd.read_csv(args.csv,float_precision='round_trip'),dict(provider='user_local_csv',path=str(args.csv.resolve()),file_sha256=digest(raw)))
    elif args.command=='record':r=record(journal)
    elif args.command=='settle':r=settle(journal)
    else:
        check_freeze();events=journal.events();r=summarize(events,utc())
        if r['status']=='PRIMARY_READY':
            existing=[e for e in events if e['kind']=='primary_evaluation']
            if existing:r=existing[0]['payload']
            else:journal.append('primary_evaluation','fixed_52_slots',r)
    print(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
