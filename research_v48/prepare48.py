from common48 import *
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'preparation_manifest.json').exists();assert len(list(ROOT.glob('*.py')))==10
    r=manifest('preparation');r['old_evidence']=old_evidence();assert read(PREV/'results/verification.json')['status']=='PASS';save(OUT/'initial_freeze.json',r);files=[OUT/'initial_freeze.json']
    for src,name in COPIES:p=OUT/name;p.write_bytes(src.read_bytes());files.append(p)
    for src,dst in [('source_heads.json','uniform_heads.json'),('heads.json','weighted_heads.json'),('source_models.json','source_models.json'),('source_testing_features.json','source_testing_features.json'),('training_weights.csv','training_weights.csv'),('weight_summary.csv','weight_summary.csv')]:
        p=OUT/dst;p.write_bytes((PREV/'results'/src).read_bytes());files.append(p)
    finish(r,files,old_files=6143,new_neural_fits=0,new_head_fits=0,assembled_head_records=144);print('R48:10sources,protocol,inputs and6143oldfiles frozen.',flush=True)
if __name__=='__main__':main()
