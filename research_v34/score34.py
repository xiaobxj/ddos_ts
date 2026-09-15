from common34 import *

def main():
    check_frozen();check_phase('fitting');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs,_,_=data();base=csv('baseline_model_predictions')
    candidate,sources=candidate_predictions(obs,read(OUT/'policy_heads.json'),read(OUT/'feature_banks.json'),csv('routing'),base);models=pd.concat([base,candidate],ignore_index=True);ensemble=ensemble_from(models)
    assert len(models)==47872 and len(ensemble)==17952;pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    files=[]
    for n,g in [('model_predictions',models),('ensemble_predictions',ensemble),('prediction_sources',sources)]:p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_model_rows=13056,ensemble_rows=17952,histories=11);print('All three frozen coefficient steps scored, with eight unchanged baselines.',flush=True)
if __name__=='__main__':main()
