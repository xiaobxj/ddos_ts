from common30 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run)
    obs,_,_=data();cover=coverage(obs);missing=extensions(cover);cover.to_csv(OUT/'coverage.csv',index=False);missing.to_csv(OUT/'extension_membership.csv',index=False)
    models=[r for r in read(V28/'results/models.json') if r['history']=='rolling5'];heads=[h for h in read(V28/'results/heads.json') if h['history']=='rolling5'];assert len(models)==69 and len(heads)==276
    save(OUT/'source_models.json',models);save(OUT/'source_heads.json',heads);base,e=baselines();base.to_csv(OUT/'baseline_model_predictions.csv',index=False);e.to_csv(OUT/'baseline_ensemble_predictions.csv',index=False)
    files=[OUT/n for n in ['initial_freeze.json','coverage.csv','extension_membership.csv','source_models.json','source_heads.json','baseline_model_predictions.csv','baseline_ensemble_predictions.csv']]
    finish(run,files,old_files=4600,coverage_model_weeks=len(cover),missing_model_weeks=len(missing),extension_networks=int(missing.cutoff.nunique())*3,extension_model_rows=len(missing)*16,new_neural_fits=0,new_head_fits=0,new_scoring=False,trigger_simulation=False)
    print(f'Frozen: {len(cover)} universal model-weeks, {len(missing)} missing model-weeks, {missing.cutoff.nunique()*3} extension network passes; no trigger simulation.',flush=True)
if __name__=='__main__':main()
