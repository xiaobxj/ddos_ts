from common37 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for name in ['annual_heads.json','feature_banks.json','routing.csv','state_thresholds.json','state_observations.csv','membership_roles.csv','weekly_context.csv']:
        p=OUT/name;p.write_bytes((V36/'results'/name).read_bytes());files.append(p)
    for source,target in [('model_predictions','baseline_model_predictions'),('ensemble_predictions','baseline_ensemble_predictions')]:p=OUT/f'{target}.csv';p.write_bytes((V36/'results'/f'{source}.csv').read_bytes());files.append(p)
    assert read(V36/'results/verification.json')['status']=='PASS';assert cfg()['policies']==POLICIES
    finish(run,files,old_files=5315,new_predictions=0,new_fits=0);print('Frozen R37 protocol, source, inputs and5315old files.',flush=True)
if __name__=='__main__':main()
