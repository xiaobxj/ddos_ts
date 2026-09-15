from common38 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for name in ['training_inputs.json','correction_heads.json','correction_parameters.csv','routing.csv','state_thresholds.json','state_observations.csv','weekly_context.csv','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','primary_comparisons.json','seed_correction_effects.csv','weekly_correction_effects.csv']:
        p=OUT/name;p.write_bytes((V37/'results'/name).read_bytes());files.append(p)
    assert read(V37/'results/verification.json')['status']=='PASS';finish(run,files,old_files=5428,new_fits=0,new_predictions=0);print('R38 source, protocol, inputs and5428old files frozen.',flush=True)
if __name__=='__main__':main()
