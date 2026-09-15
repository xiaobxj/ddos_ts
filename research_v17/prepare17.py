from common17 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (ROOT/'protocol.json').exists() and not (OUT/'preparation_manifest.json').exists(),'Preserve frozen attempt'
    old=previous.cfg()
    p=dict(version=17,experiment='Training-only 1%-99% feature winsorization plus causal market-state diagnostics',
        scope='Authorized research-only continuation. All261Friday observations have been inspected in earlier rounds. Design selected after round16 findings; per-fold causality is not an independent holdout. No new data, live actions, trading-profitability test, neural training or claim of full paper replication.',
        folds=old['folds'],windows=old['windows'],data=old['data'],seeds=old['seeds'],
        features='Reuse all24 hash-verified round16 training and validation feature caches:18 frozen MSE20 decoded representations plus6 deterministic raw25 representations. Same25 coordinates and sources. Retain unmodified native and two probe references exactly. No new representation or target.',
        clipping=dict(quantiles=[.01,.99],quantile_method='linear',description='Each coordinate uses its own annual training 1st/99th percentile. Clip training features; compute training mean and population SD after clipping; use those same bounds and moments on future features. Refit all24 logistic heads with the unchanged objective. No percentile/regularization/threshold grid and no probability clipping for prediction.'),
        standardization=old['standardization'],probe=old['probe'],independent_solver=old['independent_solver'],
        market_states=dict(purpose='Diagnostic labels only; states do not enter any classifier, calibration, weighting, threshold, date screening or model switching this round.',
            descriptors={'trend60':'log(C_t/C_(t-60)) divided by max(populationSD of60 close log returns,1e-8)*sqrt(60)',
                'volatility20':'Population SD of last20 daily close log returns',
                'range20':'Mean of log(H_s/L_s) over last20 sessions, including signal date',
                'volume_change20':'log(1+V_t)-log(1+V_(t-20))'},
            thresholds='Fit separately at each cutoff using exactly its matured-label training input rows, without reading the label values: trend terciles 1/3 and2/3; volatility/range/volume medians. Freeze thresholds through the next calendar year. Use <= for the lower cell at a boundary. Annual labels are relative to that fold training distribution, not universal bull/bear labels.',
            primary_partition='Six cells: low/mid/high trend score crossed with low/high20-session volatility.',
            secondary_partitions='Volatility, range and volume-change binary partitions separately. No high-dimensional cross-product, hidden-state smoothing, clustering search or forward-return state definitions.',
            partitions=PARTITIONS,minimum_descriptive_cell_n=20,sparse='Always report every prespecified cell and its count. Cells with n<20 are marked sparse and are not grounds for selecting an expert. One-class cells have undefined balanced accuracy and AUROC. No subgroup significance tests or state-selected aggregate.'),
        budget='24 new deterministic Newton fits:18 learned_clip and6 raw25_clip. Complete all24 before any new held-out score. Reuse18 neural states with zero neural training steps.1044new seed/head probability forecasts;1827retained model forecasts;2871model rows and1827ensemble rows=7methods*261dates. Three probabilities averaged for learned_clip; raw25_clip is one head per fold, seed=-1 is a deterministic identifier. Independent solution replay is verification, not another candidate.',
        forecasts='Strict probability>0.5; average probabilities across inherited learned seeds. Native MSE retains its average-return sign and has no probability metrics. No state-conditioned model fitted this round.',
        metrics=old['metrics'],pooled='261weeks pooled only for descriptive metrics. All comparisons and state analyses use separate120-week early and141-week late windows. Report six years and learned/native seed results.',
        primary_comparisons_per_window=[dict(candidate=c,reference=r,metric=m) for c,r,m in [
            ('learned_clip','learned_probe','brier'),('learned_clip','learned_probe','direction_error'),
            ('raw25_clip','raw25_probe','brier'),('raw25_clip','raw25_probe','direction_error'),
            ('learned_clip','training_frequency','brier'),('raw25_clip','training_frequency','brier')]],
        inference=old['inference'],
        descriptive_screen='Eight strict flags per candidate per window: accuracy beats its unmodified probe, nativeMSE and trainingfrequency; Brier beats its unmodified probe and trainingfrequency; logloss beats trainingfrequency; at least2/3years beat native accuracy; at least2/3years beat frequency Brier. All8 in bothwindows required for a cross-period descriptive pass. Ties fail. No automatic promotion; all dates previously examined.',
        shift_diagnostics='Per-head training and validation coordinate clipping fractions, validation row clipping fractions, before/after maximum standardized distance using clipped training moments, and per-state diagnostics. New clipped bounds do not reconstruct missing training regimes and may remove signal; no causal attribution from error subgroup differences.',
        verification='Freeze and preserve2716old evidence files. Verify scalar descriptor formulas, future/truncated-path invariance, cutoff threshold invariance, quantile order statistics, exact maturity/date masks, and labels absent from state fitting. Independently solve all24 heads with L-BFGS-B; replay new forecasts, old reference equality, metric/confusion/AUROC calculations and12block contrasts. Replay18 inherited neural representations and6raw representations from the original inputs; no weight changes. Recompute all state assignments and nonempty state metrics, including sparse cells.')
    save(ROOT/'protocol.json',p)
    evidence=old_evidence();assert read(V16/'results/verification.json')['status']=='PASS'
    run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv']:
        target=OUT/name;shutil.copyfile(V16/'results'/name,target);files.append(target)
    heads=read(V16/'results/heads.json');assert len(heads)==24
    save(OUT/'source_heads.json',heads);files.append(OUT/'source_heads.json')
    finish(run,files,old_evidence=evidence,new_candidate_fits=24,new_fits_performed=0,new_forecasts_performed=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),previous_files_preserved=len(evidence),new_fits=0,new_forecasts=0)),flush=True)

if __name__=='__main__':main()
