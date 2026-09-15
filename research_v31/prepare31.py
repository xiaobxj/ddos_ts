from common31 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run)
    obs,_,_=data();cover=previous_round.coverage(obs);cover.to_csv(OUT/'coverage.csv',index=False);unique=archive_predictions();unique.to_csv(OUT/'universal_model_predictions.csv',index=False)
    models=[r for r in read(V28/'results/models.json') if r['history']=='rolling5'];heads=[r for r in read(V28/'results/heads.json') if r['history']=='rolling5'];assert len(models)==69 and len(heads)==276
    save(OUT/'source_models.json',models);save(OUT/'source_heads.json',heads);a,b=baselines();a.to_csv(OUT/'baseline_model_predictions.csv',index=False);b.to_csv(OUT/'baseline_ensemble_predictions.csv',index=False)
    files=[OUT/n for n in ['initial_freeze.json','coverage.csv','universal_model_predictions.csv','source_models.json','source_heads.json','baseline_model_predictions.csv','baseline_ensemble_predictions.csv']]
    finish(run,files,old_files=4696,coverage_model_weeks=len(cover),source_networks=69,source_heads=276,new_neural_fits=0,new_head_fits=0,new_inference_passes=0,policy_simulation=False)
    print('Frozen R31: 4696 old files, 671 reused model-weeks; no new policy simulation.',flush=True)
if __name__=='__main__':main()
