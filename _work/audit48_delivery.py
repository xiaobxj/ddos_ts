from pathlib import Path
import hashlib,json

project=Path('D:/ddos_v3');root=project/'research_v48';out=root/'results'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=read(out/'delivery_manifest.json');prep=read(out/'preparation_manifest.json')
assert manifest['status']=='PASS'
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=out/'delivery_manifest.json'}
assert actual==set(manifest['files'])
for name,h in manifest['files'].items():assert sha(root/name)==h,name
for group in ['old_evidence','source_sha256','input_sha256']:
    for name,h in prep[group].items():assert sha(project/name)==h,(group,name)
assert sha(root/'protocol.json')==prep['protocol_sha256']==manifest['protocol_sha256']
assert len(prep['old_evidence'])==6143 and len(list(root.glob('*.py')))==10
for name in ['contract_verification.json','verification.json','visual_review.json','interpretation_checks.json']:
    record=read(out/name);assert record['status']=='PASS'
    for file,h in record.get('artifacts',{}).items():assert sha(root/file)==h,(name,file)
for phase in ['preparation','assembly','scoring','evaluation','report']:
    record=read(out/f'{phase}_manifest.json')
    for file,h in record['artifacts'].items():assert sha(root/file)==h,(phase,file)
assert not (project/'research_v49').exists()
print(json.dumps(dict(status='PASS',delivered_files=len(actual),delivery_manifest_additional=1,old_files_preserved=len(prep['old_evidence']),frozen_current_sources=10,inherited_and_current_sources=len(prep['source_sha256']),frozen_input_files=len(prep['input_sha256']),protocol_unchanged=True,all_phase_and_QA_artifact_hashes_valid=True),indent=2))
