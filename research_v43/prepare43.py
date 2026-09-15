from common43 import *

def main():
    OUT.mkdir(exist_ok=True); assert not (OUT/'preparation_manifest.json').exists()
    run=manifest('preparation'); assert len(list(ROOT.glob('*.py')))==8
    run['old_evidence']=old_evidence(); assert read(PREV/'results/verification.json')['status']=='PASS'
    save(OUT/'initial_freeze.json',run); files=[OUT/'initial_freeze.json']
    for n in COPIES:
        p=OUT/n;p.write_bytes((PREV/'results'/n).read_bytes());files.append(p)
    finish(run,files,old_files=5754,new_fits=0,new_policies=0)
    print('R43 8 sources, protocol, inputs and 5754 old files frozen.',flush=True)
if __name__=='__main__': main()
