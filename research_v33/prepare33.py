from common33 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for source,target in [('source_heads.json','annual_heads.json'),('heads.json','quarter_heads.json'),('feature_banks.json','feature_banks.json'),('primary_comparisons.json','baseline_comparisons.json')]:p=OUT/target;save(p,read(V32/'results'/source));files.append(p)
    for source,target in [('routing','routing'),('membership','head_training_membership'),('ensemble_metrics','baseline_metrics'),('yearly_metrics','baseline_yearly_metrics'),('model_predictions','model_predictions'),('ensemble_predictions','ensemble_predictions')]:
        g=pd.read_csv(V32/'results'/f'{source}.csv',float_precision='round_trip')
        if source in ['model_predictions','ensemble_predictions']:g=g[g.history.isin([cfg()['reference'],cfg()['candidate']])&g.method.isin(METHODS)].reset_index(drop=True)
        p=OUT/f'{target}.csv';g.to_csv(p,index=False);files.append(p)
    finish(run,files,old_files=4895,weeks=272,methods=4,new_neural_fits=0,new_head_fits=0,new_feature_inference=0,new_decomposition=False);print('Frozen R33 sources and read-only diagnostic scope; no new decomposition.',flush=True)
if __name__=='__main__':main()
