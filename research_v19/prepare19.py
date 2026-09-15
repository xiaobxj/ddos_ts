from common19 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (ROOT/'protocol.json').exists() and not (OUT/'preparation_manifest.json').exists(),'Preserve frozen attempt'
    old=previous.cfg();independent=dict(old['independent_solver']);independent['purpose']='Independently verify all24new heads; never use alternate coefficients for forecasts.'
    p=dict(version=19,experiment='One residualized volatility-by-inherited-signal interaction per annual additive head',
        scope='Authorized research-only continuation. All261test dates already examined; this design follows round18 diagnostics. Causal annual training and forecasting do not create an independent holdout. No new data, deployment, profitability test or paper replication claim.',
        folds=old['folds'],windows=old['windows'],data=old['data'],seeds=old['seeds'],
        variants=[dict(method='learned_vol_interaction',parent='learned_market',base_dimensions=29,dimensions=30,fits=18),
            dict(method='raw_vol_interaction',parent='raw_trend',base_dimensions=26,dimensions=27,fits=6)],
        interaction=dict(volatility='Reuse the exact round18training-clipped/standardized volatility20 coordinate, market index1, frozen separately each cutoff.',
            signal='r=X[:,0:25] @ parent.coefficients[0:25]. This is the representation-only logit contribution of the already fitted round18parent. It excludes that parent appended market coefficients and intercept. Raw25itself already contains three market summaries. Ray weights are supervised training-fit coefficients, not out-of-fold predictions; no next-year labels are used.',
            product='q=v*r. Only one fixed product, no separate25-dimensional products, higher powers, horizon grid or expert models.',
            projection='Using only this fold training inputs, solve min_gamma ||q-[X,1]gamma||^2 with numpy.linalg.lstsq rcond=1e-12. This removes the product component linearly representable by existing additive coordinates. Freeze gamma, training residual mean and populationSD (floor1e-6). h=(q-[X,1]gamma-residual_mean)/residual_sd; append h to unchanged X.',
            normalization='No additional clipping of product or residual. Constituent inputs retain their existing training-only clipping. Do not recompute projection, means, SDs or ray on held-out data.',
            model='Refit the full ridge logistic head on [X,h], one extra slope. All additive slopes and intercept may refit; interaction ray stays fixed. Zero new slope nests the original parent exactly. Same lambda=.01 applies to every slope, intercept unpenalized.',
            interpretation='This is a one-dimensional interaction along an inherited signal direction, with training-linear redundancy removed. Residualization changes the ridge parameterization relative to an unresidualized-product model; that alternative is not tested. Coefficient signs/components are not causal importance or OOF calibration evidence.',
            sd_floor=1e-6,projection_rcond=1e-12,rank_tolerance=1e-8,training_orthogonality_tolerance=1e-8),
        probe=old['probe'],independent_solver=independent,
        projection_verification=dict(method='scipy.linalg.lstsq, pivoted-QR gelsy',rcond=1e-12,training_fitted_value_absolute_tolerance=1e-9,
            training_standardized_residual_absolute_tolerance=1e-8,validation_standardized_residual_absolute_tolerance=1e-8),
        budget='24new deterministic CPU float64 classifier fits:18learned,6raw. One SVD projection per head is an input transformation, not another classifier candidate. Complete all24heads before any new held-out scoring.1044new model probabilities; retain4176round18model records. Total5220model records,3132ensemble records=12methods*261dates. No neural forward/training. Preserve2882old evidence files.',
        forecasts='Same strict p>0.5. Average3inherited learned-seed probabilities; raw has one deterministic head per cutoff,seed=-1. Retain all10round18methods unchanged. NativeMSE is a return score, not probability.',
        metrics=old['metrics'],pooled=old['pooled'],
        primary_comparisons_per_window=[dict(candidate=c,reference=r,metric=m) for c,r,m in [
            ('learned_vol_interaction','learned_market','brier'),('learned_vol_interaction','learned_market','direction_error'),
            ('raw_vol_interaction','raw_trend','brier'),('raw_vol_interaction','raw_trend','direction_error'),
            ('learned_vol_interaction','learned_clip','brier'),('raw_vol_interaction','raw25_clip','brier'),
            ('learned_vol_interaction','training_frequency','brier'),('raw_vol_interaction','training_frequency','brier')]],
        inference='16prespecified contrasts=8in each period; candidate-minus-reference loss,negativebetter. Same circular8retained-observation blocks,10000replicates,seed20260910 separately perwindow. Centered two-sided p; Holm across16. No pooled primary test. Retained blocks can cross calendar gaps. Intervals do not cover model/design selection or repeated research.',
        descriptive_screen='For each candidate perwindow use10strict flags: accuracy exceeds direct round18parent,round17anchor,nativeMSE,trainingfrequency; Brier below parent,round17anchor,trainingfrequency; logloss below trainingfrequency; at least2/3years accuracy exceeds nativeMSE; at least2/3years Brier below trainingfrequency. Tiesfail. Bothwindows all10 required; no automatic promotion.',
        diagnostics='Report all previous state partitions with n<20marked sparse; no subgroup tests or state-selected aggregate. Record projection rank, residual SD, orthogonality, interaction coefficients and train/validation contribution mean/SD plus direction changes. A coefficient change is not an isolated causal intervention. No posthoc feature-removal forecasts.',
        verification='Freeze protocol/core/input hashes. Verify inherited joins, source ray, training-only projection and future-transform invariance. Independent QR reconstruction of all24SVD transforms, Newton replay and independent L-BFGS-B solutions. Independently verify new/unchanged predictions, all metric groups and16block contrasts. Neural provenance relies on unchanged round17backbone replay and round18cache/descriptor verification; do not claim new neural replay.')
    save(ROOT/'protocol.json',p);evidence=old_evidence();assert read(V18/'results/verification.json')['status']=='PASS'
    run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv','training_states.csv','validation_states.csv']:
        target=OUT/name;shutil.copyfile(V18/'results'/name,target);files.append(target)
    sources=[h for h in read(V18/'results/heads.json') if h['method'] in CANDIDATES.values()];assert len(sources)==24;jobs=[]
    for s in sources:
        variant=next(v for v in p['variants'] if v['parent']==s['method'])
        jobs.append(dict(job=f"{variant['method']}_{s['cutoff']}_{s['seed']}",method=variant['method'],source_job=s['job'],cutoff=s['cutoff'],seed=s['seed'],
            parent_method=s['method'],base_dimensions=s['dimensions'],dimensions=variant['dimensions']))
    save(OUT/'source_heads.json',sources);save(OUT/'jobs.json',jobs);files += [OUT/'source_heads.json',OUT/'jobs.json']
    finish(run,files,old_evidence=evidence,new_primary_fits_performed=0,new_predictions_performed=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence),new_fits=0,new_predictions=0)),flush=True)

if __name__=='__main__':main()
