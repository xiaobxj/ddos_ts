from common46 import *

def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');files=[]
    for name in ['correction_heads.json','correction_parameters.csv']:
        p=OUT/name;p.write_bytes((PREV/'results'/name).read_bytes());files.append(p)
    heads=read(OUT/'correction_heads.json');assert len(heads)==1104 and sum(h['fit_eligible'] for h in heads)==588
    check_frozen();finish(run,files,reused_scalar_fits=588,head_records=1104,new_scalar_fits=0,new_neural_fits=0,validation_labels_used=False)
    print('R45 coefficients reused byte-for-byte:588 fitted scalar offsets,1104 head records; no new fitting.',flush=True)
if __name__=='__main__':main()
