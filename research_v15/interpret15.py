"""Descriptive summaries after frozen evaluation; no candidate fitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    train=pd.read_csv(OUT/'training_metrics.csv')
    for target in ['log_loss','brier']:
        train[target+'_skill']=1-train[target]/train['constant_'+target]
        train['native_'+target+'_skill']=1-train['native_'+target]/train['constant_'+target]
    fields=['train_n','iterations','objective','penalty','log_loss','constant_log_loss','native_log_loss','brier','constant_brier','native_brier',
        'accuracy','native_accuracy','auroc','probability_std','native_probability_std','log_loss_skill','brier_skill','native_log_loss_skill','native_brier_skill']
    train.groupby(['family','cutoff'])[fields].mean().reset_index().to_csv(OUT/'training_fold_summary.csv',index=False)
    seed=pd.read_csv(OUT/'all_seed_predictions.csv');ensemble=pd.read_csv(OUT/'all_ensemble_predictions.csv');rows=[]
    for method in ['direction_bce','probe_mse','probe_bce']:
        g=seed[seed.method.eq(method)].pivot(index='date',columns='seed',values='probability')
        h=ensemble[ensemble.method.eq(method)].set_index('date').loc[g.index]
        p=g.to_numpy();y=h.actual_up.to_numpy()[:,None];mean=p.mean(axis=1)
        seed_brier=float(np.square(p-y).mean());ensemble_brier=float(np.square(mean-y[:,0]).mean())
        dispersion=float(p.var(axis=1).mean());assert abs(seed_brier-ensemble_brier-dispersion)<1e-14
        np.testing.assert_allclose(mean,h.probability,rtol=0,atol=1e-14)
        rows.append(dict(method=method,mean_seed_brier=seed_brier,ensemble_brier=ensemble_brier,
            mean_seed_probability_variance=dispersion,identity_max_error=abs(seed_brier-ensemble_brier-dispersion)))
    pd.DataFrame(rows).to_csv(OUT/'ensemble_brier_decomposition.csv',index=False)
    print('Descriptive training summaries and exact Brier ensemble decomposition saved.')


if __name__=='__main__':main()
