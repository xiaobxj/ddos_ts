from common41 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    assert read(V40/'results/verification.json')['status']=='PASS'
    for src,name in COPIES:
        dest=OUT/name;dest.write_bytes(src.read_bytes());files.append(dest)
    finish(run,files,old_files=5633,new_fits=0,new_predictions=0);print('R41 read-only inputs,8sources,protocol and5633oldfiles frozen.',flush=True)
if __name__=='__main__':main()
