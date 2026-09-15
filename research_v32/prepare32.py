from common32 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);obs,_,_=data();jobs,members,route=memberships(obs);banks=bank_memberships(obs,members);files=[]
    for name,g in [('jobs',jobs),('membership',members),('routing',route),('bank_membership',banks),('budgets',runtime_budgets())]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    for name,records in [('source_models',model_sources()),('source_heads',head_sources()),('source_testing_features',test_sources())]:p=OUT/f'{name}.json';save(p,records);files.append(p)
    a,b=baselines()
    for name,g in [('baseline_model_predictions',a),('baseline_ensemble_predictions',b)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    files.append(OUT/'initial_freeze.json');finish(run,files,old_files=4759,new_neural_fits=0,new_head_jobs=51,new_head_fits=204,new_projections=102,bank_rows_per_seed=len(banks),missing_rows_per_seed=int(banks.source.eq(2).sum()),bank_extension_passes=18,new_scoring=False)
    print(f'Frozen R32: 51 quarter-seed head jobs; {int(banks.source.eq(2).sum())} missing daily feature rows per seed; no new fitting.',flush=True)
if __name__=='__main__':main()
