"""Isolated integration tests after the frozen R50 engineering replay passes."""
from pathlib import Path
import sys,tempfile,copy
from unittest.mock import patch
ROOT=Path('D:/ddos_v3/research_v50');sys.path.insert(0,str(ROOT))
from common50 import *
from build50 import activate,registered,select_package,refresh
from quarter50 import Engine,extend_bank
from data49 import load_snapshot
from run50 import record,status
def rejected(call):
    try:call()
    except (ValueError,FileExistsError):return
    raise AssertionError('Expected rejection missing')
check_freeze(True);initial=Journal().events();bootstrap_path=PROJECT/'research_v49/results/bootstrap_package.json';bootstrap=read(bootstrap_path);cases=[]
with tempfile.TemporaryDirectory(prefix='r50_lifecycle_',dir=PROJECT/'_work') as directory:
    directory=Path(directory).resolve();assert directory.is_relative_to((PROJECT/'_work').resolve())
    j=Journal(directory/'journal');j.initialize('synthetic')
    receipt=directory/'bootstrap_receipt.json';save(receipt,dict(status='PASS',package_sha256=sha(bootstrap_path)))
    event=activate(j,bootstrap_path,receipt);assert event['scope']=='synthetic' and len(j.events())==1
    chosen,p=select_package(j,'2026-09-18');assert p==bootstrap and chosen==event
    rejected(lambda:select_package(j,'2026-10-09'));rejected(lambda:activate(j,bootstrap_path,receipt))
    late=Journal(directory/'late');late.initialize('synthetic')
    with patch.object(late,'events',return_value=[dict(kind='prediction',key='2026-09-18')]):rejected(lambda:activate(late,bootstrap_path,receipt))
    badreceipt=directory/'failed.json';save(badreceipt,dict(status='FAIL',package_sha256=sha(bootstrap_path)));rejected(lambda:activate(late,bootstrap_path,badreceipt))
    replay=copy.deepcopy(bootstrap);replay['scope']='engineering_replay';replaypath=directory/'replay.json';save(replaypath,replay)
    replayreceipt=directory/'replay_receipt.json';save(replayreceipt,dict(status='PASS',package_sha256=sha(replaypath)));rejected(lambda:activate(late,replaypath,replayreceipt))
    # A changed receipt must invalidate even a previously registered package.
    receipt.write_bytes(encoded(dict(status='PASS',package_sha256=sha(bootstrap_path),tampered=True)));rejected(lambda:registered(j))
    assert late.events()==[]
cases+=['verified_bootstrap_registration_and_exact_date_routing','wrong_quarter_and_duplicate_registration_rejected','replacement_after_forecast_rejected','failed_receipt_and_engineering_package_rejected','registered_receipt_hash_tamper_rejected']
snapshot=next(e for e in reversed(initial) if e['kind']=='snapshot');frame=load_snapshot(Journal(),snapshot)
engine=Engine(bootstrap);date='2026-08-28';forecast=engine.predict(frame[frame.date<=date],date,production=False)
# This object never enters a journal. It exercises preference for a previously
# committed forecast while keeping the real journal unchanged.
fixture=dict(kind='prediction',key=date,recorded_utc='2026-08-28T12:00:00+00:00',payload=dict(forecast=forecast))
extended=extend_bank(frame,snapshot['payload']['end'],bootstrap,{bootstrap['annual_cutoff']:bootstrap},initial+[fixture])
selected=extended[extended.date.eq(date)];assert len(selected)==12 and selected.calibration_source.eq('committed_prospective_prediction').all()
for row in selected.itertuples():
    original=next(p for p in forecast['seed_predictions'] if p['history']==U and p['method']==row.method and p['seed']==row.seed)
    assert row.probability==original['probability'] and row.annual_logit==original['logit']
assert extended[extended.date.eq('2026-09-04')].calibration_source.eq('calibration_only_replay_not_prospective').all()
cases+=['committed_U_predictions_preferred_without_refitting','missed_week_calibration_replay_remains_separate']
before_files=artifacts(PARAMS) if PARAMS.exists() else {}
rejected(lambda:refresh(Journal(),'2026-09-30'));rejected(lambda:refresh(Journal(),'2026-12-31'));rejected(lambda:record(Journal()))
assert (artifacts(PARAMS) if PARAMS.exists() else {})==before_files and Journal().events()==initial
cases+=['real_future_refresh_rejected_before_any_build_output','real_out_of_window_record_rejected_without_journal_mutation']
check_freeze(True)
save(OUT/'lifecycle_verification.json',dict(status='PASS',completed_utc=iso(utc()),cases=cases,synthetic_journals_isolated_and_removed=True,real_journal_unchanged=True,future_build_files_created=0,actual_prospective_predictions=sum(e['kind']=='prediction' for e in initial),verification_sha256=sha(OUT/'verification.json'),freeze_sha256=sha(OUT/'freeze.json')))
print(encoded(dict(status='PASS',integration_checks=len(cases),future_build_files_created=0)).decode())
