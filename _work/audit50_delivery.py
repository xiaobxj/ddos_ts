from pathlib import Path
import hashlib,json
project=Path('D:/ddos_v3');root=project/'research_v50';out=root/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
d=read(out/'delivery_manifest.json');freeze=read(out/'freeze.json');assert d['status']=='PASS'
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=out/'delivery_manifest.json'}
assert actual==set(d['files'])
for n,h in d['files'].items():assert sha(root/n)==h,n
for group in ['old_evidence','immutable_files']:
    for n,h in freeze[group].items():assert sha(project/n)==h,(group,n)
assert sha(root/'protocol.json')==d['protocol_sha256']==freeze['protocol_sha256']
assert sha(out/'freeze.json')==d['freeze_sha256']
for name in ['verification.json','lifecycle_verification.json','interpretation_checks.json']:
    r=read(out/name);assert r['status']=='PASS'
    for n,h in r.get('artifacts',{}).items():assert sha(project/n)==h,(name,n)
runtime=read(out/'runtime_handoff_manifest.json')
for n,h in runtime['files'].items():assert sha(project/n)==h,n
paths=sorted((project/'prospective_r49/events').glob('*.json'));assert len(paths)==2
previous=sha(project/'prospective_r49/genesis.json')
for i,path in enumerate(paths,1):
    e=read(path);assert e['sequence']==i and e['previous_sha256']==previous
    for n,h in e['blobs'].items():assert sha(project/'prospective_r49/blobs'/n)==h
    previous=sha(path)
    assert e['kind'] in ['snapshot','parameter_package']
e=read(paths[-1]);p=e['payload'];assert sha(project/p['package_file'])==p['package_sha256'] and sha(project/p['verification_file'])==p['verification_sha256']
receipt=read(project/p['verification_file']);assert receipt['status']=='PASS' and receipt['package_sha256']==p['package_sha256']
assert not (project/'prospective_r49/writer.lock').exists() and not (project/'prospective_r50').exists()
archive=project/'research_v50_attempt1';preserved=read(archive/'preservation.json')
for n,h in preserved['files'].items():assert sha(archive/n)==h
assert len(list(archive.rglob('*.py')))==9 and len([p for p in archive.rglob('*') if p.is_file()])==13
assert len(freeze['old_evidence'])==6230 and len(list(root.glob('*.py')))==9
print(json.dumps(dict(status='PASS',delivered_files=len(actual),additional_delivery_manifest=1,old_files_preserved=6230,archived_preflight_files=13,final_source_files=9,registered_bootstrap_packages=1,prospective_predictions=0,future_parameter_builds=0),indent=2))
