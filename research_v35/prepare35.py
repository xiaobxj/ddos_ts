from common35 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for name in ['source_models.json','source_heads.json','source_testing_features.json','feature_banks.json']:
        p=OUT/name;save(p,read(V34/'results'/name));files.append(p)
    for source,target in [('endpoints.json','both_heads.json'),('training_designs.json','both_training_designs.json'),('primary_comparisons.json','baseline_comparisons.json')]:p=OUT/target;save(p,read(V34/'results'/source));files.append(p)
    for n in ['jobs','membership','routing','bank_membership']:
        p=OUT/f'{n}.csv';p.write_bytes((V34/'results'/f'{n}.csv').read_bytes());files.append(p)
    for n,g in zip(['baseline_model_predictions','baseline_ensemble_predictions'],baselines()):p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    for n,g in zip(['membership_summary','ablation_membership','set_changes'],membership_tables()):p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    assert read(V34/'results/verification.json')['status']=='PASS';finish(run,files,old_files=5063,new_fit_jobs=102,new_head_fits=408,weekly_scoring=False)
    print('R35 frozen: four member sets,408 new fits; full-roll and annual sources reused.',flush=True)
if __name__=='__main__':main()
