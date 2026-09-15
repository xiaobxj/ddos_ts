"""Create, verify and activate immutable parameter packages in the original journal."""
from common50 import *
from data49 import load_snapshot,unchanged_prefix
from annual50 import fit_annual
from quarter50 import extend_bank,fit_quarter
from validate50 import validate_annual,validate_quarter,validate_package_schema
from datetime import datetime

def registered(journal):
    result=[]
    for event in journal.events():
        if event['kind']!='parameter_package':continue
        item=event['payload'];guard(item['implementation_freeze_sha256']==sha(OUT/'freeze.json'),'Unsupported parameter implementation')
        path=checked_path(item['package_file']);guard(sha(path)==item['package_sha256'],'Registered package changed');p=read(path)
        vp=checked_path(item['verification_file']);guard(sha(vp)==item['verification_sha256'],'Package verification changed');v=read(vp)
        guard(v['status']=='PASS' and v['package_sha256']==sha(path),'Package not verified')
        validate_package_schema(p);result.append((event,p))
    return result
def select_package(journal,date):
    matches=[(e,p) for e,p in registered(journal) if p['quarter_cutoff']==quarter_for(date) and p['annual_cutoff']==annual_for(date)]
    guard(len(matches)==1,'No unique verified package for this annual/quarter cutoff')
    e,p=matches[0];guard(p['valid_from']<=date<=p['valid_until'],'Expired package');return e,p
def activate(journal,package_path,verification_path):
    check_freeze(True);p=read(package_path);v=read(verification_path);validate_package_schema(p)
    guard(v['status']=='PASS' and v['package_sha256']==sha(package_path),'Unverified package cannot activate')
    guard(p.get('scope')!='engineering_replay','Replay package cannot activate')
    guard(not any(e['kind']=='prediction' and p['valid_from']<=e['key']<=p['valid_until'] for e in journal.events()),'Cannot replace parameters after predictions in this quarter')
    return journal.append('parameter_package',p['quarter_cutoff'],dict(package_file=relative(package_path),package_sha256=sha(package_path),verification_file=relative(verification_path),verification_sha256=sha(verification_path),implementation_freeze_sha256=sha(OUT/'freeze.json'),annual_cutoff=p['annual_cutoff'],quarter_cutoff=p['quarter_cutoff'],valid_until=p['valid_until']))
def request_check(cutoff,snapshot_event,at):
    allowed={quarter_for(d) for d in cohort()['signal_slots']};guard(cutoff in allowed and cutoff>='2026-09-30','Unsupported refresh cutoff')
    boundary=datetime.fromisoformat(cutoff+'T18:00:00+08:00')
    guard(at>=boundary,'Cutoff data is not yet available; future refit forbidden')
    guard(datetime.fromisoformat(snapshot_event['recorded_utc'])>=boundary and snapshot_event['payload']['end']>=cutoff,'Snapshot is too old for requested cutoff')
def refresh(journal,cutoff):
    check_freeze(True);events=journal.events();snapshots=[e for e in events if e['kind']=='snapshot'];guard(snapshots,'No snapshot')
    source=snapshots[-1];request_check(cutoff,source,utc());guard(not any(e['kind']=='parameter_package' and e['key']==cutoff for e in events),'Quarter package already activated')
    choices=registered(journal);expected_previous=(pd.Timestamp(cutoff).to_period('Q').start_time-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    parents=[(e,p) for e,p in choices if p['quarter_cutoff']==expected_previous];guard(len(parents)==1,'Missing immediately preceding quarter package');_,parent=parents[0]
    full=load_snapshot(journal,source)
    if 'source_snapshot_sequence' in parent:
        prior_event=events[parent['source_snapshot_sequence']-1];prior=load_snapshot(journal,prior_event);unchanged_prefix(prior,full,prior.date.iloc[-1])
    # No downstream training or label builder receives rows beyond cutoff.
    frame=full[full.date.le(cutoff)].reset_index(drop=True)
    directory=PARAMS/'builds'/f'{cutoff}_{uuid.uuid4().hex}';directory.mkdir(parents=True,exist_ok=False)
    save(directory/'request.json',dict(cutoff=cutoff,started_utc=iso(utc()),snapshot_sequence=source['sequence'],snapshot_event_sha256=sha(journal.root/'events'/f'{source["sequence"]:06d}.json'),implementation_freeze_sha256=sha(OUT/'freeze.json')))
    csv_save(directory/'cutoff_prefix.csv',frame)
    try:
        if cutoff.endswith('12-31'):
            annual=fit_annual(frame,cutoff,directory/'annual');annual_check=validate_annual(frame,annual)
        else:
            annual={k:copy.deepcopy(parent[k]) for k in ['annual_cutoff','models','uniform','weighted','volatility_median','return_scale','training_up_frequency']};annual_check=dict(status='PASS',mode='unchanged_annual_package',parent_quarter=parent['quarter_cutoff'])
        packages={p['annual_cutoff']:p for _,p in choices};packages[annual['annual_cutoff']]=annual
        bank=extend_bank(frame,cutoff,parent,packages,events);csv_save(directory/'calibration_bank.csv',bank)
        quarterly=fit_quarter(bank,cutoff);qcheck=validate_quarter(bank,cutoff,quarterly);save(directory/'quarter_fit.json',quarterly)
        # Verify all calibration labels against the cutoff-only joint target table.
        from dataset50 import observations
        _,obs,_=observations(frame,cutoff);truth=obs.set_index('date')
        for date,g in bank.groupby('date'):
            guard(date in truth.index and g.actual_up.eq(int(truth.loc[date,'exec_return']>0)).all() and g.joint_completed.eq(truth.loc[date,'joint_completed']).all(),'Calibration labels disagree with cutoff prefix')
        dependencies=dict(parent['dependencies']);dependencies.update(annual.get('annual_artifacts',{}));dependencies.update(artifacts(directory))
        package=dict(annual,package_id=f'annual{annual["annual_cutoff"]}_quarter{cutoff}',scope='prospective',assembled_utc=iso(utc()),quarter_cutoff=cutoff,valid_from=(pd.Timestamp(cutoff)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),valid_until=quarter_end(cutoff),quarter_offsets=quarterly['quarter_offsets'],calibration_bank=relative(directory/'calibration_bank.csv'),source_snapshot_sequence=source['sequence'],source_snapshot_sha256=sha(journal.root/'events'/f'{source["sequence"]:06d}.json'),dependencies=dependencies,implementation_freeze_sha256=sha(OUT/'freeze.json'))
        path=directory/'package.json';save(path,package);validate_package_schema(package)
        verification=directory/'verification.json';save(verification,dict(status='PASS',completed_utc=iso(utc()),package_sha256=sha(path),annual=annual_check,quarter=qcheck,cutoff_prefix_maximum_date=frame.date.iloc[-1],calibration_labels_verified=len(bank),source_snapshot_sha256=package['source_snapshot_sha256']))
        check_freeze(True);return activate(journal,path,verification)
    except Exception as exc:
        save(directory/'failure.json',dict(status='FAILED_NOT_ACTIVATED',completed_utc=iso(utc()),error_type=type(exc).__name__,error=str(exc)));raise
