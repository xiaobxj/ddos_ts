"""Preserve and repair only the elapsed-time local-variable shadow in R27 verification.

Frozen research_v27/verify27.py and all training/scoring/evaluation code remain
unchanged. Execute its body with the calibration lower-bound variable renamed.
All verification assertions and tolerances are unchanged and rerun in full.
"""
from pathlib import Path
import sys,json,hashlib,time
PROJECT=Path(__file__).resolve().parent.parent;ROOT=PROJECT/'research_v27';OUT=ROOT/'results'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')
prep=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
source_file=ROOT/'verify27.py';original=source_file.read_text(encoding='utf-8')
assert sha(source_file)==prep['source_sha256'][str(source_file.relative_to(PROJECT))]
before="cutoff=fold['cutoff'];start=f'{int(cutoff[:4])-2}-01-01';g=pred"
after="cutoff=fold['cutoff'];calibration_start=f'{int(cutoff[:4])-2}-01-01';g=pred"
assert original.count(before)==1 and original.count('pred.date.ge(start)')==1
repaired=original.replace(before,after).replace('pred.date.ge(start)','pred.date.ge(calibration_start)')
archive=OUT/'verification_attempt1';archive.mkdir(exist_ok=False)
files={}
for name in ['independent_solutions.csv','checkpoint_replays.csv','logs/verify.log']:
    source=OUT/name;target=archive/source.name;target.write_bytes(source.read_bytes());files[target.name]=sha(target)
save(archive/'preservation.json',dict(reason='All36checkpoint and144solver checks and metrics completed; final elapsed_seconds write failed because calibration window start string shadowed initial time variable.',files=files,new_model_training_fits=0,new_scored_candidates=0))
(OUT/'verification_metadata_adapter.txt').write_bytes(Path(__file__).read_bytes())
evidence=dict(status='RUNNING',repair='Rename only calibration lower-bound local variable start to calibration_start; no setting,condition,tolerance,fit,prediction or statistical change.',original_verifier_sha256=sha(source_file),executed_body_sha256=hashlib.sha256(repaired.encode('utf-8')).hexdigest(),adapter_sha256=sha(__file__),adapter_copy='verification_metadata_adapter.txt',failed_attempt='verification_attempt1/preservation.json',primary_neural_fits_unchanged=36,primary_head_fits_unchanged=144,actual_verification_checkpoint_replays_including_failed_attempt=72,actual_independent_solver_runs_including_failed_attempt=288)
save(OUT/'verification_metadata_repair.json',evidence);started=time.time()
sys.path.insert(0,str(ROOT));namespace={'__name__':'__main__','__file__':str(source_file)}
exec(compile(repaired,str(source_file),'exec'),namespace)
for name in ['independent_solutions.csv','checkpoint_replays.csv']:assert sha(OUT/name)==files[name],name
evidence.update(status='PASS',elapsed_seconds=time.time()-started,independent_results_bitwise_equal_to_first_attempt=True)
save(OUT/'verification_metadata_repair.json',evidence)
print('Metadata repair PASS: full verification repeated; solver and replay tables byte-identical to preserved first attempt.',flush=True)
