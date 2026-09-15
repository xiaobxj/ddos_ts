"""Post-result arithmetic diagnosis from frozen predictions; no fits or new forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/'results'


def main():
    data=pd.read_csv(OUT/'ensemble_predictions.csv');rows=[]
    for name,g in data.groupby('schedule',sort=False):
        y=g.actual.to_numpy(float);p=g.predicted_return.to_numpy(float)
        bias=float((p.mean()-y.mean())**2)
        varp=float(p.var());vary=float(y.var())
        covariance=float(np.mean((p-p.mean())*(y-y.mean())))
        mse=float(np.mean((p-y)**2));total=bias+varp+vary-2*covariance
        assert abs(mse-total)<1e-15
        rows.append(dict(schedule=name,n=len(g),prediction_mean=float(p.mean()),actual_mean=float(y.mean()),
            squared_bias=bias,prediction_variance=varp,target_variance=vary,prediction_target_covariance=covariance,
            negative_twice_covariance=-2*covariance,mse=mse,reconstructed_mse=total))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'mse_moment_decomposition.csv',index=False)
    indexed=frame.set_index('schedule');base=indexed.loc['archived20'];changes=[]
    for name in ['constant40','decay40']:
        r=indexed.loc[name]
        row=dict(candidate=name,reference='archived20',mse_change=r.mse-base.mse)
        for key in ['squared_bias','prediction_variance','negative_twice_covariance','target_variance']:
            row[key+'_change']=float(r[key]-base[key])
        assert abs(sum(row[k+'_change'] for k in ['squared_bias','prediction_variance','negative_twice_covariance','target_variance'])-row['mse_change'])<1e-15
        changes.append(row)
    pd.DataFrame(changes).to_csv(OUT/'mse_change_decomposition.csv',index=False)
    result=dict(status='PASS',timing='Post-result arithmetic diagnosis, separate from the frozen primary comparisons',
        identity='MSE = squared mean bias + prediction variance + target variance - 2*population covariance',
        definitions='Population moments over the same141 pooled weeks; units are raw squared return. This is an accounting identity, not causal attribution.',
        model_states=3,paired_changes=2,additional_fitting=False,new_predictions=False,
        calibration_fitted_on_validation=False,changes=changes)
    (OUT/'moment_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(frame.to_string(index=False));print(pd.DataFrame(changes).to_string(index=False))


if __name__=='__main__':main()
