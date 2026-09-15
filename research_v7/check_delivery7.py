"""Read-only final checksum, numerical metric and inference-family audit."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    delivery=read('delivery_manifest.json')
    for name,digest in delivery['files'].items():
        with (ROOT/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
    report=(ROOT/delivery['report']).read_text(encoding='utf-8')
    assert report.startswith('# 第七轮方法测试：') and '\ufffd' not in report
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if not target.startswith(('https://','http://')):assert (OUT/target).resolve().exists(),target
    audit=read('verification.json');assert audit['status']=='PASS' and audit['total_checkpoints_replayed']==144
    seed=pd.read_csv(OUT/'all_seed_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');comparison=pd.read_csv(OUT/'candidate_comparisons.csv')
    assert len(seed)==6768 and len(ensemble)==2256
    definitions={'candidate20':('combined',20),'baseline10':('baseline',10),
                 'baseline20':('baseline',20),'combined10':('combined',10)}
    for name,(arm,epoch) in definitions.items():
        g=seed[seed.role==name]
        assert len(g)==1692 and g.arm.eq(arm).all() and g.epoch.eq(epoch).all()
    for r in metrics.itertuples():
        g=ensemble[(ensemble.role==r.role)&(ensemble.history==r.history)]
        assert len(g)==141
        mse=float(np.mean((g.predicted_return-g.actual)**2))
        assert abs(mse-r.mse)<1e-12
        assert abs(np.sqrt(mse)-r.rmse)<1e-12
        assert abs(((g.predicted_return>0)==(g.actual>0)).mean()-r.accuracy)<1e-12
        assert abs(float(np.mean((g.reassigned_prediction-g.actual)**2))-r.reassigned_mse)<1e-12
    flags={'absolute_advantage':True,'relative_advantage':True,'seed_consistency':True}
    for r in comparison.itertuples():
        g=ensemble[(ensemble.role=='candidate20')&(ensemble.history==r.history)]
        b=ensemble[(ensemble.role=='baseline10')&(ensemble.history==r.history)]
        candidate=float(np.mean((g.predicted_return-g.actual)**2))
        mean=float(np.mean((g.training_mean-g.actual)**2))
        baseline=float(np.mean((b.predicted_return-b.actual)**2))
        assert abs(candidate**.5-r.candidate_rmse)<1e-12
        assert abs(1-candidate/mean-r.mse_skill_vs_mean)<1e-12
        assert abs(1-candidate/baseline-r.mse_skill_vs_baseline10)<1e-12
        count=0
        for number,h in seed[(seed.role=='candidate20')&(seed.history==r.history)].groupby('seed'):
            count+=int(np.mean((h.predicted_return-h.actual)**2)<np.mean((h.training_mean-h.actual)**2))
        assert count==r.candidate_seeds_better_than_mean
        if r.history!='full':
            flags['absolute_advantage'] &= candidate<mean
            flags['relative_advantage'] &= candidate<baseline
            flags['seed_consistency'] &= count>=2
    assessment=read('stability_assessment.json')
    assert flags==assessment['flags'] and all(flags.values())==assessment['consistency_screen_pass']
    pairs=read('primary_comparisons.json');assert len(pairs)==6
    probabilities=[]
    for r in pairs:
        g=ensemble[(ensemble.role=='candidate20')&(ensemble.history==r['history'])].sort_values('date')
        if r['reference']=='training_mean':
            b=g.training_mean.to_numpy()
        else:
            h=ensemble[(ensemble.role==r['reference'])&(ensemble.history==r['history'])].sort_values('date')
            assert h.date.tolist()==g.date.tolist()
            b=h.predicted_return.to_numpy()
        y=g.actual.to_numpy();a=g.predicted_return.to_numpy()
        values=(a-y)**2-(b-y)**2
        rng=np.random.default_rng(20260910)
        starts=rng.integers(len(g),size=(10000,int(np.ceil(len(g)/8))))
        ids=((starts[:,:,None]+np.arange(8))%len(g)).reshape(10000,-1)[:,:len(g)]
        delta=values.mean();bootstrap=values[ids].mean(axis=1)
        centered=(values-delta)[ids].mean(axis=1)
        p=float((1+(np.abs(centered)>=abs(delta)).sum())/10001)
        assert abs(p-r['mse']['p'])<1e-12
        assert abs(delta-r['mse']['difference'])<1e-12
        np.testing.assert_allclose(np.quantile(bootstrap,[.025,.975]),[r['mse']['ci95_low'],r['mse']['ci95_high']],rtol=0,atol=1e-12)
        probabilities.append(p)
    adjusted=np.zeros(6);running=0.
    for i,index in enumerate(np.argsort(probabilities)):
        running=max(running,min(1.,probabilities[index]*(6-i)))
        adjusted[index]=running
    np.testing.assert_allclose(adjusted,[r['mse']['holm_adjusted_p'] for r in pairs],rtol=0,atol=1e-12)
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),
                         fixed_candidate='combined20',new_fits=54,replayed_checkpoints=144,
                         matched_prefix_checkpoints=audit['prior_prefix_checkpoints_with_identical_weights'],
                         consistency_flags=flags,consistency_screen_pass=all(flags.values()),
                         primary_comparisons_recomputed=6),indent=2))


if __name__=='__main__':main()
