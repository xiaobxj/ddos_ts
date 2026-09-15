from pathlib import Path
import hashlib,json,shutil
project=Path('D:/ddos_v3').resolve();root=project/'research_v50';archive=project/'research_v50_attempt1'
assert root.resolve()==Path('D:/ddos_v3/research_v50').resolve() and not archive.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert len(files)==11 and not (root/'results/annual_replay').exists()
for name,h in files.items():
    target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target);assert sha(target)==h
diagnosis=dict(status='PREFLIGHT_FAILED_BEFORE_TRAINING',failure='Exact return target scale parity assertion',raw_return_mean_gap=2.3852447794681098e-18,raw_return_sd_gap=3.469446951953614e-17,maximum_raw_return_gap=9.974659986866641e-17,diagnostic='Original R5 returns were obtained by pandas default CSV parsing of R4 observation_table; reapplying this exact CSV serialization/parsing stage reproduces the archived target vector bit-for-bit. Aux targets, packed inputs, float32 standardized labels and membership already matched.',new_neural_fits=0,new_head_fits=0,registered_parameters=0)
(archive/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')
(archive/'preservation.json').write_text(json.dumps(dict(status='PRESERVED',original_root=str(root),files=files),indent=2),encoding='utf-8')
for name in ['protocol.json','results/freeze.json']:
    p=root/name;assert sha(p)==sha(archive/name);p.unlink()
print('Preserved all11original frozen files plus diagnosis/preservation; no training or activation had started.')
