from common43 import *
def main():
    prep=check_frozen()
    for n in ['verification.json','visual_review.json','interpretation_checks.json']:
        r=read(OUT/n);assert r['status']=='PASS'
        for name,h in r.get('artifacts',{}).items():assert sha(ROOT/name)==h,name
    for phase in ['diagnosis','report']:check_phase(phase)
    assert old_evidence()==prep['old_evidence'];path=OUT/'delivery_manifest.json';assert not path.exists()
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=path]
    save(path,dict(status='PASS',completed_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},old_files_preserved=5754))
    print(f'Delivery PASS: {len(files)} files plus manifest.',flush=True)
if __name__=='__main__':main()
