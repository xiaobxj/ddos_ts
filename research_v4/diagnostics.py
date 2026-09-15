"""Descriptive post-run prediction/seed/fit diagnostics; no model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent;out=root/'results'
config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
pred=pd.read_csv(out/'predictions.csv');curves=pd.read_csv(out/'training_curves.csv')
rows=[]
for arm in config['arms']:
    g=pred[pred.method==arm['name']]
    total=float(((g.predicted_return-g.predicted_return.mean())**2).sum())
    within=sum(float(((h.predicted_return-h.predicted_return.mean())**2).sum()) for _,h in g.groupby('cutoff'))
    for cutoff,h in g.groupby('cutoff'):
        c=curves[(curves.phase=='development')&(curves.method==arm['name'])&(curves.cutoff==cutoff)]
        rows.append(dict(method=arm['name'],cutoff=cutoff,n=len(h),
                         mean_forecast=float(h.predicted_return.mean()),
                         within_fold_forecast_sd=float(h.predicted_return.std(ddof=0)),
                         within_fold_actual_sd=float(h.actual.std(ddof=0)),
                         long_weeks=int(h.position.sum()),
                         between_fold_share_of_forecast_variation=1-within/total if total else None,
                         seed_mean_first_epoch_training_loss=float(c[c.epoch==1].train_mse_standardized.mean()),
                         seed_mean_last_epoch_training_loss=float(c[c.epoch==5].train_mse_standardized.mean())))
pd.DataFrame(rows).to_csv(out/'prediction_variation_diagnostics.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
