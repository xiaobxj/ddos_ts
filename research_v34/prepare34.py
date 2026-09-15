from common34 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for name in ['source_models.json','source_heads.json','source_testing_features.json','feature_banks.json']:
        p=OUT/name;save(p,read(V32/'results'/name));files.append(p)
    for name in ['jobs','membership','routing','bank_membership']:
        p=OUT/f'{name}.csv';p.write_bytes((V32/'results'/f'{name}.csv').read_bytes());files.append(p)
    for name,g in zip(['baseline_model_predictions','baseline_ensemble_predictions'],baselines()):
        p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    p=OUT/'baseline_comparisons.json';save(p,read(V32/'results/primary_comparisons.json'));files.append(p)
    assert read(V33/'results/verification.json')['status']=='PASS' and read(V32/'results/verification.json')['status']=='PASS'
    finish(run,files,old_files=4946,new_neural_fits=0,new_feature_inference=0,new_transform_fits=0,new_endpoint_fits=204,derived_policy_heads=612,weekly_scoring=False)
    print('R34 frozen: annual transforms, primary25%, sensitivity50%, control100%; no new fit yet.',flush=True)
if __name__=='__main__':main()
