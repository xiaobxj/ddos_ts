from pathlib import Path
import sys
PROJECT=Path('D:/ddos_v3')
sys.path.insert(0,str(PROJECT/'research_v51'))
from common51 import *

freeze=check_freeze(True)
delivery=read(OUT/'delivery_manifest.json')
assert delivery['status']=='PASS' and delivery['freeze_sha256']==sha(OUT/'freeze.json')
for name,digest in delivery['files'].items():
    assert sha(ROOT/name)==digest,name
assert len(delivery['files'])==16 and old_evidence()==freeze['old_evidence']
for section in ['real_runtime_files','operational_receipt_files']:
    for name,digest in delivery[section].items():
        assert sha(PROJECT/name)==digest,name
v=read(OUT/'verification.json'); c=read(OUT/'cli_acceptance.json')
assert v['status']==c['status']=='PASS' and len(v['cases'])==22 and len(c['checks'])==7
replay=read(OUT/'synthetic_rehearsal.json')
assert sha(OUT/'synthetic_rehearsal.json')==v['synthetic_rehearsal_sha256']
fixture=(PROJECT/v['synthetic_fixture_directory']/'end_to_end').resolve()
assert fixture.is_relative_to((PROJECT/'_work').resolve())
for name,digest in replay['fixture_files'].items():
    assert sha(fixture/name)==digest,name
assert read(fixture/'SYNTHETIC_TEST_ONLY_journal/genesis.json')['synthetic_test_only'] is True
events=Journal().events()
assert len(events)==2 and not any(e['kind'] in ['prediction','label'] for e in events)
assert not (RUNTIME/'cycle.lock').exists() and not (previous.base.RUNTIME/'writer.lock').exists()
print(encoded(dict(status='PASS',delivery_files=17,preserved_old_files=6279,
    engineering_checks=22,cli_checks=7,synthetic_artifact_hashes=len(replay['fixture_files']),
    real_journal_events=2,real_predictions=0,real_labels=0)).decode('utf-8'))
