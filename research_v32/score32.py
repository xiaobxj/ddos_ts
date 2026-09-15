from common32 import *
def main():
    check_frozen();check_phase('fitting');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs,_,_=data();base=csv('baseline_model_predictions');candidate,provenance=candidate_predictions(obs,read(OUT/'heads.json'),read(OUT/'feature_banks.json'),csv('routing'),base);models=pd.concat([base,candidate],ignore_index=True);ensemble=ensemble_from(models);assert len(models)==34816 and len(ensemble)==13056
    pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0);files=[]
    for name,g in [('model_predictions',models),('ensemble_predictions',ensemble),('prediction_sources',provenance)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_model_rows=4352,ensemble_rows=13056,source_heads=204,reused_annual_heads=72);print('Scoring PASS: annual neural features with quarterly heads and seven unchanged baselines.',flush=True)
if __name__=='__main__':main()
