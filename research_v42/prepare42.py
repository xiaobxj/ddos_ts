from common42 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json'];assert read(V41/'results/verification.json')['status']=='PASS'
    for src,name in COPIES:
        p=OUT/name;p.write_bytes(src.read_bytes());files.append(p)
    finish(run,files,old_files=5691,new_fits=0,new_predictions=0);print('R42 sources,protocol,inputs and5691oldfiles frozen.',flush=True)
if __name__=='__main__':main()
