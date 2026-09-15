from common39 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for src,dst in [('model_predictions.csv','baseline_model_predictions.csv'),('ensemble_predictions.csv','baseline_ensemble_predictions.csv'),('ensemble_metrics.csv','baseline_metrics.csv'),('routing.csv','routing.csv'),('weekly_context.csv','weekly_context.csv'),('seed_correction_effects.csv','source_seed_effects.csv')]:
        p=OUT/dst;p.write_bytes((V38/'results'/src).read_bytes());files.append(p)
    assert read(V38/'results/verification.json')['status']=='PASS';finish(run,files,old_files=5484,new_fits=0,new_predictions=0);print('R39 source, protocol, inputs and5484old files frozen.',flush=True)
if __name__=='__main__':main()
