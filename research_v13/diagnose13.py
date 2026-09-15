"""Post-result accounting only: frozen-prediction sign changes, no fitted alternatives."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    assert not (OUT/'direction_diagnostic.json').exists(),'Preserve diagnostic evidence'
    predictions=pd.read_csv(OUT/'outer_ensemble_predictions.csv')
    baseline=predictions[predictions.method.eq('archived20')].set_index('date')
    rows=[];thresholds=[]
    for method in ['rolling_shrink','half_shrink']:
        g=predictions[predictions.method.eq(method)].set_index('date')
        np.testing.assert_array_equal(g.index,baseline.index)
        for year in ['all','2018','2019','2020']:
            h=g if year=='all' else g[g.index.str.startswith(year)]
            b=baseline.loc[h.index];y=h.actual.to_numpy()>0;p=h.predicted_return.to_numpy()>0;q=b.predicted_return.to_numpy()>0
            old_correct=q==y;new_correct=p==y
            lost=int((old_correct&~new_correct).sum());gained=int((~old_correct&new_correct).sum())
            n_to_up=int((~q&p).sum());up_to_n=int((q&~p).sum())
            assert lost+gained==n_to_up+up_to_n
            assert int(new_correct.sum())==int(old_correct.sum())-lost+gained
            rows.append(dict(method=method,year=year,n=len(h),changed=n_to_up+up_to_n,
                correct_to_wrong=lost,wrong_to_correct=gained,non_up_to_up=n_to_up,up_to_non_up=up_to_n,
                original_correct=int(old_correct.sum()),candidate_correct=int(new_correct.sum()),
                mse_skill_vs_original=float(1-np.mean((h.predicted_return-h.actual)**2)/np.mean((b.predicted_return-b.actual)**2))))
        for cutoff,h in g.groupby('cutoff'):
            a=float(h.alpha.iloc[0]);m=float(h.training_mean.iloc[0]);assert m>0
            threshold=None if a==0 else m*(1-1/a)
            b=baseline.loc[h.index];expected=np.ones(len(h),dtype=bool) if a==0 else b.predicted_return.to_numpy()>threshold
            np.testing.assert_array_equal(expected,h.predicted_return.to_numpy()>0)
            thresholds.append(dict(method=method,cutoff=cutoff,alpha=a,training_mean=m,
                original_prediction_up_threshold=threshold,all_up_due_to_zero_alpha=a==0))
    pd.DataFrame(rows).to_csv(OUT/'direction_changes.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'implied_direction_thresholds.csv',index=False)
    result=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),post_result_diagnostic=True,
        additional_fitting=False,new_predictions=False,outer_optimal_alpha_estimated=False,
        identity='For alpha>0, m+alpha*(p-m)>0 iff p>m*(1-1/alpha). All outer training means are positive.',
        input_sha256=sha(OUT/'outer_ensemble_predictions.csv'),
        files={n:sha(OUT/n) for n in ['direction_changes.csv','implied_direction_thresholds.csv']})
    (OUT/'direction_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
