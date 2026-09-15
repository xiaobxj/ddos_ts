"""Post-evaluation descriptions of feature shift and fixed-logit accounting.

These diagnostics do not refit, clip, select or replace any prediction.
"""
from common16 import *


def main():
    check_frozen();check_phase('evaluation');assert read(OUT/'verification.json')['status']=='PASS'
    refs={r['job']:r for r in read(OUT/'validation_features.json')};pred=pd.read_csv(OUT/'model_predictions.csv')
    columns=[];summary=[]
    for head in read(OUT/'heads.json'):
        d=load_features(head);ref=refs[head['job']];assert sha(PROJECT/ref['cache_file'])==ref['cache_sha256']
        with np.load(PROJECT/ref['cache_file']) as a:f=a['features'].astype(float)
        x=(f-d['mean'])/d['sd'];theta=np.asarray(head['coefficients']);z=design(x)@theta
        contributions=x.mean(axis=0)*theta[:-1]
        assert abs(z.mean()-theta[-1]-contributions.sum())<1e-12
        low=d['standardized'].min(axis=0);high=d['standardized'].max(axis=0);outside=((x<low)|(x>high)).mean(axis=0)
        for j in range(25):
            columns.append(dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],feature=j,
                feature_name=NAMES[j] if head['kind']=='raw' else f'decoded_{j}',coefficient=float(theta[j]),
                validation_standardized_mean=float(x[:,j].mean()),validation_standardized_sd=float(x[:,j].std()),
                mean_logit_contribution=float(contributions[j]),outside_training_range_fraction=float(outside[j])))
        g=pred[pred.method.eq(head['method'])&pred.cutoff.eq(head['cutoff'])&pred.seed.eq(head['seed'])].sort_values('row_index')
        np.testing.assert_allclose(z,g.logit,rtol=0,atol=1e-12)
        summary.append(dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],test_n=len(f),
            training_up_frequency=float(d['direction'].mean()),validation_up_frequency=float(g.actual_up.mean()),
            training_intercept=float(theta[-1]),mean_validation_logit=float(z.mean()),sum_feature_logit_contributions=float(contributions.sum()),
            mean_validation_probability=float(probability(z).mean()),maximum_absolute_standardized_mean=float(np.abs(x.mean(axis=0)).max()),
            mean_outside_training_range_fraction=float(outside.mean()),identity_error=abs(float(z.mean()-theta[-1]-contributions.sum()))))
    pd.DataFrame(columns).to_csv(OUT/'posthoc_feature_shift.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'posthoc_logit_accounting.csv',index=False)
    train=pd.read_csv(OUT/'training_metrics.csv');train['log_loss_skill']=1-train.log_loss/train.constant_log_loss
    train.groupby(['method','cutoff'])[['train_n','accuracy','auroc','log_loss','constant_log_loss','brier','constant_brier','log_loss_skill']].mean().reset_index().to_csv(OUT/'training_fold_summary.csv',index=False)
    save(OUT/'posthoc_diagnostic_provenance.json',dict(completed_utc=now(),stage='after_frozen_evaluation_and_verification',
        fixed_models=True,new_predictions=0,new_fits=0,causal_explanation_claim=False,
        description='Accounting identity: mean validation logit equals trained intercept plus the sum of coefficient times mean standardized validation features. Feature means use the frozen training scaler. This describes fixed-model behavior; it neither identifies causal market drivers nor tests a new policy.'))
    print(pd.DataFrame(summary).query("method=='raw25_probe'").to_string(index=False),flush=True)


if __name__=='__main__':main()
