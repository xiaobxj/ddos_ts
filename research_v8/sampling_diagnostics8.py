"""Training-only diagnostics of retained labels, quarter mass and batch reduction.

Nominal loss coefficients are not gradients or actual AdamW parameter influence.
This analysis does not change any model, seed, sample or validation comparison.
"""
from common8 import *


def main():
    check_frozen();obs=pd.read_csv(OUT/'observation_table.csv')
    with np.load(V5/'cache/targets.npz') as data:yall=data['returns'].astype(float)
    rows=[];exposures=[];quarters=[]
    for fold in cfg()['folds']:
        for policy in cfg()['policies']:
            tr,w,_,n=load_policy(fold['cutoff'],policy);y=yall[tr]
            mean=np.average(y,weights=w);var=np.average((y-mean)**2,weights=w)
            left=obs.entry.iloc[tr].to_numpy();right=np.maximum(obs.exit.iloc[tr],obs.anchor.iloc[tr]+5).to_numpy()+1
            adjacent=(left[1:]<right[:-1])
            rows.append(dict(cutoff=fold['cutoff'],policy=policy,train_n=len(tr),full_n=n,return_mean=float(mean),return_sd=float(var**.5),
                adjacent_retained_return_correlation=float(np.corrcoef(y[:-1],y[1:])[0,1]),adjacent_joint_label_overlap_fraction=float(adjacent.mean()),
                last_batch_size=n%128 or 128,steps_per_epoch=int(np.ceil(n/128)),repetitions_per_epoch=n/len(tr)))
            q=pd.to_datetime(obs.date.iloc[tr]).dt.to_period('Q').astype(str).to_numpy()
            for key in sorted(set(q)):
                selected=q==key
                quarters.append(dict(cutoff=fold['cutoff'],policy=policy,quarter=key,rows=int(selected.sum()),
                    weight_mass=float(w[selected].sum()/w.sum()),
                    weighted_mean_loss_share=float(np.sum(w[selected]*(y[selected]-mean)**2)/np.sum(w*(y-mean)**2))))
            for seed in cfg()['training']['seeds']:
                rng=np.random.default_rng(seed);draws=np.zeros(len(tr),dtype=int);coefficients=np.zeros(len(tr));last_rows=np.zeros(len(tr),dtype=int)
                for _ in range(20):
                    order=legacy.epoch_order(len(tr),n,rng)
                    np.add.at(draws,order,1)
                    for lo in range(0,n,128):
                        b=order[lo:lo+128];np.add.at(coefficients,b,w[b]/len(b))
                    np.add.at(last_rows,order[(int(np.ceil(n/128))-1)*128:],1)
                nominal=coefficients/coefficients.sum();intended=w/w.sum();ratios=nominal/intended
                exposures.append(dict(cutoff=fold['cutoff'],policy=policy,seed=seed,unique_n=len(tr),presentations=20*n,
                    minimum_draws=int(draws.min()),maximum_draws=int(draws.max()),minimum_last_batch_count=int(last_rows.min()),
                    maximum_last_batch_count=int(last_rows.max()),nominal_loss_coefficient_ratio_min=float(ratios.min()),
                    nominal_loss_coefficient_ratio_max=float(ratios.max()),nominal_loss_coefficient_ratio_sd=float(ratios.std()),
                    interpretation='Coefficient of sample loss before clipping/AdamW; not actual parameter influence.'))
    pd.DataFrame(rows).to_csv(OUT/'sampling_label_diagnostics.csv',index=False)
    pd.DataFrame(exposures).to_csv(OUT/'nominal_batch_exposure.csv',index=False)
    pd.DataFrame(quarters).to_csv(OUT/'weighted_quarter_loss.csv',index=False)
    save(OUT/'sampling_diagnostics_manifest.json',dict(training_only=True,changed_training=False,
        purpose='Interpretation of frozen policies and inherited final partial batches; no validation-dependent intervention.',
        source_sha256=sha(Path(__file__)),completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print(pd.DataFrame(rows)[['cutoff','policy','train_n','adjacent_retained_return_correlation','last_batch_size']].to_string(index=False))


if __name__=='__main__':main()
