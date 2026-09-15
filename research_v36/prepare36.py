from common36 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for source,target in [('source_heads.json','annual_heads.json'),('feature_banks.json','feature_banks.json'),('primary_comparisons.json','baseline_comparisons.json')]:p=OUT/target;save(p,read(V35/'results'/source));files.append(p)
    for source,target in [('routing','routing'),('model_predictions','model_predictions'),('ensemble_predictions','ensemble_predictions'),('ensemble_metrics','baseline_metrics'),('yearly_metrics','baseline_yearly_metrics'),('weekly_effects','archived_weekly_effects')]:p=OUT/f'{target}.csv';p.write_bytes((V35/'results'/f'{source}.csv').read_bytes());files.append(p)
    for n,g in zip(['membership_roles','required_rows'],memberships()):p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    assert read(V35/'results/verification.json')['status']=='PASS';finish(run,files,old_files=5250,new_head_fits=0,new_neural_fits=0,new_feature_inference=0,new_predictions=0,new_state_calibration=False);print('R36 protocol and read-only diagnostic inputs frozen.',flush=True)
if __name__=='__main__':main()
