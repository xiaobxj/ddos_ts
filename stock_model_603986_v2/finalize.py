"""Seal only a fully verified, rendered and inspected research delivery."""
from core import *

def main():
    check_freeze()
    for n in ['models_verified.json','outputs_verified.json','controls_verified.json','scoring_completed.json']:
        assert read(OUT/n)['status']=='PASS',n
    for p,s in read(OUT/'evaluation_implementation_freeze.json')['files'].items():assert sha(ROOT/p)==s,p
    for p,s in read(OUT/'control_verifier_dtype_audit.json')['files'].items():assert sha(ROOT/p)==s,p
    for p,s in read(OUT/'preserved_files.json').items():assert sha(PROJECT/p)==s,p
    assert len(list((OUT/'fits').glob('*/model_*_e20.pt')))==84
    assert len(list((OUT/'fits').glob('*/model_*_e10.pt')))==84
    assert len(read(OUT/'prior_problem_coverage.json'))==51
    assert (ROOT/'快慢模型测试.png').is_file() and (ROOT/'快慢模型测试报告.md').is_file()
    save(OUT/'delivery_manifest.json',dict(status='PASS',completed_utc=now(),training_freeze_sha256=sha(OUT/'freeze.json'),
        scoring_freeze_sha256=sha(OUT/'evaluation_implementation_freeze.json'),
        neural_fits=84,checkpoints=168,prior_rounds_mapped=51,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in str(p)}))
    print('Delivery sealed; original research and stock v1 preserved.')

if __name__=='__main__':main()
