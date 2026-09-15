from common46 import *
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'preparation_manifest.json').exists();assert len(list(ROOT.glob('*.py')))==11
    r=manifest('preparation');r['old_evidence']=old_evidence();assert read(PREV/'results/verification.json')['status']=='PASS';save(OUT/'initial_freeze.json',r);files=[OUT/'initial_freeze.json']
    for src,name in COPIES:p=OUT/name;p.write_bytes(src.read_bytes());files.append(p)
    finish(r,files,old_files=5986,new_neural_fits=0);print('R46 11 sources, protocol, inputs and5986oldfiles frozen.',flush=True)
if __name__=='__main__':main()
