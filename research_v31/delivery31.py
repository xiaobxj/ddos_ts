from common31 import *
def main():
    prep=check_frozen();assert read(OUT/'verification.json')['status']=='PASS';assert read(OUT/'visual_review.json')['status']=='PASS'
    for phase in ['scoring','evaluation','report']:check_phase(phase)
    assert old_evidence()==prep['old_evidence'];path=OUT/'delivery_manifest.json';assert not path.exists()
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=path]
    save(path,dict(status='PASS',completed_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},old_files_preserved=len(prep['old_evidence'])))
    print(f'Delivery PASS:{len(files)}files plus manifest.',flush=True)
if __name__=='__main__':main()
