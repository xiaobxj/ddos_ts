"""Manual R49 cohort runner with validated quarterly/annual parameter routing."""
import argparse
from common50 import *
from build50 import refresh,select_package,registered
from quarter50 import Engine
from data49 import fetch,snapshot,load_snapshot
from run49 import settle
from evaluate49 import summarize
from datetime import datetime

def status(journal):
    check_freeze(True);events=journal.events();at=utc();date=at.astimezone(TZ).strftime('%Y-%m-%d');future=[d for d in cohort()['signal_slots'] if d>=date]
    entries=registered(journal);current=[p for _,p in entries if p['valid_from']<=date<=p['valid_until']]
    return dict(status='WAIT_FOR_SIGNAL_WINDOW' if not record_window(date,at) else 'SIGNAL_WINDOW_OPEN',now_utc=iso(at),next_scheduled_friday=future[0] if future else None,active_packages=len(entries),current_package=current[0]['package_id'] if len(current)==1 else None,next_parameter_cutoff=current[0]['valid_until'] if len(current)==1 else quarter_for(date),latest_snapshot=max([e['payload']['end'] for e in events if e['kind']=='snapshot'],default=None),predictions=sum(e['kind']=='prediction' for e in events),labels=sum(e['kind']=='label' for e in events),scheduler_installed=False)
def record(journal):
    check_freeze(True);at=utc();date=at.astimezone(TZ).strftime('%Y-%m-%d');guard(date in cohort()['signal_slots'] and record_window(date,at),'Not a current eligible recording window; no backfill')
    events=journal.events();guard(not any(e['kind']=='prediction' and e['key']==date for e in events),'Duplicate prediction')
    sources=[e for e in events if e['kind']=='snapshot' and e['payload']['end']==date and datetime.fromisoformat(e['recorded_utc']).astimezone(TZ).strftime('%Y-%m-%d')==date];guard(sources,'No same-day final-bar snapshot')
    source=sources[-1];entry,package=select_package(journal,date);guard(datetime.fromisoformat(entry['recorded_utc'])<=at,'Package registration is in the future')
    result=Engine(package).predict(load_snapshot(journal,source),date);check_freeze(True)
    guard(sha(checked_path(entry['payload']['package_file']))==entry['payload']['package_sha256'],'Package changed during inference')
    return journal.append('prediction',date,dict(snapshot_sequence=source['sequence'],snapshot_sha256=sha(journal.root/'events'/f'{source["sequence"]:06d}.json'),package_file=entry['payload']['package_file'],package_sha256=entry['payload']['package_sha256'],package_registration_sequence=entry['sequence'],protocol_sha256=sha(PROJECT/'research_v49/protocol.json'),implementation_freeze_sha256=sha(OUT/'freeze.json'),forecast=result))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['status','fetch','ingest','refresh','record','settle','evaluate']);p.add_argument('--cutoff');p.add_argument('--csv',type=Path);args=p.parse_args();j=Journal();check_freeze(True)
    if args.command=='status':r=status(j)
    elif args.command=='fetch':r=fetch(j)
    elif args.command=='ingest':
        guard(args.csv is not None,'--csv required');r=snapshot(j,load_csv(args.csv),dict(provider='user_local_csv',file_sha256=sha(args.csv),path=str(args.csv.resolve())))
    elif args.command=='refresh':guard(args.cutoff is not None,'--cutoff required');r=refresh(j,args.cutoff)
    elif args.command=='record':r=record(j)
    elif args.command=='settle':r=settle(j)
    else:
        events=j.events();old=[e for e in events if e['kind']=='primary_evaluation'];r=old[0]['payload'] if old else summarize(events,utc())
        if not old and r['status']=='PRIMARY_READY':j.append('primary_evaluation','fixed_52_slots',r)
    print(encoded(r).decode('utf-8'))
if __name__=='__main__':main()
