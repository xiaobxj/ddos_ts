"""Additional read-only arithmetic audits: finite-draw variability and legacy loss parity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def main():
    draws=pd.concat([pd.read_csv(OUT/'archived_loss_draws.csv'),pd.read_csv(OUT/'restart_loss_draws.csv')])
    pairs=pd.read_csv(OUT/'paired_training_comparisons.csv')
    rows=[]
    for r in pairs[pairs['mode'].eq('dropout')].itertuples():
        mask=draws.cutoff.eq(r.cutoff)&draws.seed.eq(r.seed)&draws['mode'].eq('dropout')
        left=draws[mask&draws.state.eq(r.left_state)].set_index('draw').sort_index()
        right=draws[mask&draws.state.eq(r.right_state)].set_index('draw').sort_index()
        assert left.index.tolist()==right.index.tolist()==list(range(8))
        np.testing.assert_array_equal(left.audit_seed,right.audit_seed)
        row=dict(comparison=r.comparison,cutoff=r.cutoff,seed=int(r.seed),draws=8)
        for key in ['return_huber','auxiliary_mse','common_joint']:
            d=left[key]-right[key]
            row.update({key+'_difference_mean':float(d.mean()),key+'_difference_draw_sd':float(d.std(ddof=1)),
                        key+'_difference_min':float(d.min()),key+'_difference_max':float(d.max()),
                        key+'_negative_draws':int(d.lt(0).sum())})
            assert abs(d.mean()-getattr(r,key+'_difference'))<1e-11
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'paired_dropout_variability.csv',index=False)
    # Old round10 values used a float32 sum path; tolerate only float-rounding differences.
    prior=pd.read_csv(ROOT.parent/'research_v10/results/matched_training_objectives.csv')
    losses=pd.read_csv(OUT/'training_loss_summary.csv')
    maximum=0.
    for r in prior.itertuples():
        for state,column in [('archived_mse','mse_model_under_huber_joint'),('archived_huber','huber_model_under_huber_joint')]:
            g=losses[losses.state.eq(state)&losses.cutoff.eq(r.cutoff)&losses.seed.eq(r.seed)&losses['mode'].eq('eval')].iloc[0]
            delta=abs(g.common_joint-getattr(r,column)); maximum=max(maximum,delta)
            assert delta<2e-7
    result=dict(status='PASS',paired_dropout_variability_rows=len(rows),legacy_training_objectives_checked=18,
                maximum_legacy_loss_rounding_difference=maximum,additional_fitting=False,new_forward_passes=0)
    (OUT/'supplementary_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
