from pathlib import Path
import hashlib,json
project=Path('D:/ddos_v3');root=project/'research_v49';out=root/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
d=read(out/'delivery_manifest.json');f=read(out/'freeze.json');assert d['status']=='PASS'
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=out/'delivery_manifest.json'}
assert actual==set(d['files'])
for n,h in d['files'].items():assert sha(root/n)==h,n
for group in ['old_evidence','immutable_files']:
    for n,h in f[group].items():assert sha(project/n)==h,(group,n)
for n,h in read(out/'runtime_initial_manifest.json')['files'].items():assert sha(project/n)==h,n
assert d['protocol_sha256']==f['protocol_sha256']==sha(root/'protocol.json')
assert d['freeze_sha256']==sha(out/'freeze.json')
v=read(out/'verification.json');q=read(out/'interpretation_checks.json')
for r in [v,q]:
    assert r['status']=='PASS'
    for n,h in r['artifacts'].items():assert sha(root/n)==h,n
events=list((project/'prospective_r49/events').glob('*.json'));assert len(events)==1
e=read(events[0]);assert e['kind']=='snapshot' and e['previous_sha256']==sha(project/'prospective_r49/genesis.json')
for n,h in e['blobs'].items():assert sha(project/'prospective_r49/blobs'/n)==h,n
assert not (project/'prospective_r49/writer.lock').exists()
assert len(f['old_evidence'])==6206 and len(list(root.glob('*.py')))==7
print(json.dumps(dict(status='PASS',delivered_files=len(actual),additional_delivery_manifest=1,old_files_preserved=6206,frozen_source_files=7,initial_runtime_files=4,archived_snapshots=1,prospective_predictions=0,prospective_labels=0),indent=2))
