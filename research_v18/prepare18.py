from common18 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (ROOT/'protocol.json').exists() and not (OUT/'preparation_manifest.json').exists(),'Preserve frozen attempt'
    old=previous.cfg();probe=dict(old['probe']);probe.pop('slope_parameters');independent=dict(old['independent_solver'])
    independent['purpose']='Independently verify all30 new heads; alternate coefficients never used for forecasts.'
    protocol=dict(version=18,experiment='Continuous market context in a shared additive classifier; explicit duplicate-column control',
        scope='Authorized research-only continuation. Design follows round17 diagnostics. All261evaluated dates previously examined; causal yearly fitting is not independent confirmation. No new data, live actions, profitability test, neural training or full paper replication claim.',
        folds=old['folds'],windows=old['windows'],data=old['data'],seeds=old['seeds'],
        model_form='One annual ridge logistic classifier on concatenated standardized columns. Learned representation coefficients are shared across all market conditions; market terms shift the logit additively. No products/interactions, expert switching, cluster labels, HMM smoothing, state-dependent thresholds or outcome-based sample selection.',
        variants=[dict(method='learned_market',parent='learned_clip',representation_dimensions=25,market_indices=[0,1,2,3],dimensions=29,fits=18),
            dict(method='raw_trend',parent='raw25_clip',representation_dimensions=25,market_indices=[0],dimensions=26,fits=6),
            dict(method='market4',parent=None,representation_dimensions=0,market_indices=[0,1,2,3],dimensions=4,fits=6)],
        market_features=dict(names=MARKET_NAMES,formulas=old['market_states']['descriptors'],
            normalization='Reuse exact round17 causal descriptor values. Fit1st/99th percentile clipping bounds and then mean/populationSD from each annual training input only; floor SD at1e-6. Apply the same4-dimensional transform to all three families in that fold. All four variables derive from the existing OHLCV history; no external data or new lookback selection.',
            existing_representation='Keep every old25-dimensional round17 clipped/standardized coordinate exactly. Jointly refit its slopes and the appended slopes with the same penalty. Do not refit the existing feature normalization.',
            duplicate_control='Raw25 already includes volatility20 at index11, range20 at13 and volume_change20 at14. Do not append these exact duplicate columns because equal ridge penalties would change their effective penalty. The raw family appends only trend60, a nonlinear summary of existing60-session return/volatility. Verify raw and standardized parity for the three omitted columns; no duplicate-based model search.'),
        probe=probe,independent_solver=independent,
        budget='30 new deterministic CPU float64 primary fits:18learned_market,6raw_trend,6market4. Each head has one unpenalized intercept in addition to29/26/4slopes. Complete all30 fits before any new held-out scoring. Reuse frozen feature caches associated with18 prior neural states; no new neural forward/training pass.1305new probability records=783+261+261. Retain2871round17model records and1827ensemble records. Total4176model records,2610ensemble records=10methods*261dates.',
        forecasts='Same strict p>0.5. Average the three inherited learned-seed probabilities; raw_trend and market4 have one deterministic head per cutoff, seed=-1 is not a random replicate. Keep all7round17ensemble references unchanged, including nativeMSE with no probability metric.',
        metrics=old['metrics'],pooled=old['pooled'],
        primary_comparisons_per_window=[dict(candidate=c,reference=r,metric=m) for c,r,m in [
            ('learned_market','learned_clip','brier'),('learned_market','learned_clip','direction_error'),
            ('raw_trend','raw25_clip','brier'),('raw_trend','raw25_clip','direction_error'),
            ('learned_market','market4','brier'),('raw_trend','market4','brier'),('market4','training_frequency','brier')]],
        inference='14prespecified contrasts=7in each window. Candidate-minus-reference losses;negative better. Circular8retained-observation blocks,10000replicates,seed20260910 separately per window; centered two-sided p with Holm across all14. Retained blocks may span calendar gaps. Intervals do not cover fitting uncertainty, design selection or repeated use across rounds; no pooled primary test.',
        descriptive_screen='For each augmented candidate in eachwindow use nine strict flags: accuracy beats its direct round17parent,nativeMSE,trainingfrequency; Brier beats parent,market4,trainingfrequency; logloss beats trainingfrequency; at least2/3years beat native accuracy; at least2/3years beat frequency Brier. Bothwindows need all9. Ties fail. Market4 is a prespecified simple comparator, not automatically substituted as a selected winner. No automatic promotion.',
        diagnostics='Reuse all12state cells/partitions per window from round17, minimum20unique dates. All state metrics descriptive; no subgroup tests or model selection. Record fixed-head market logit contribution mean/SD and representation contribution mean/SD per fold, and all coefficients. Correlated contributions are not causal importance or an isolated intervention. No scoring of hypothetical feature-removal variants.',
        verification='Preserve2795old evidence files and frozen protocols. Verify train-only market transforms against independent order statistics, full raw descriptor/scalar replay, exact row joins and three duplicate-column parity. Test solver in4/26/29dimensions and extension nesting of parent objectives. Independently solve30heads with L-BFGS-B and replay every new forecast. Verify old-reference equality, all aggregate/year/seed/state metrics and14paired contrasts. Neural feature extraction provenance relies on hash-unchanged round17 PASS with18backbone replays, not a newly claimed neural replay.')
    save(ROOT/'protocol.json',protocol);evidence=old_evidence();assert read(V17/'results/verification.json')['status']=='PASS'
    run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv','training_states.csv','validation_states.csv']:
        target=OUT/name;shutil.copyfile(V17/'results'/name,target);files.append(target)
    parents=read(V17/'results/heads.json');assert len(parents)==24;jobs=[]
    for h in parents:
        method='learned_market' if h['kind']=='learned' else 'raw_trend';v=next(v for v in protocol['variants'] if v['method']==method)
        jobs.append(dict(job=f"{method}_{h['cutoff']}_{h['seed']}",method=method,cutoff=h['cutoff'],seed=h['seed'],parent_job=h['job'],
            representation_dimensions=25,market_indices=v['market_indices'],dimensions=v['dimensions']))
    for f in protocol['folds']:
        jobs.append(dict(job=f"market4_{f['cutoff']}_-1",method='market4',cutoff=f['cutoff'],seed=-1,parent_job=None,
            representation_dimensions=0,market_indices=[0,1,2,3],dimensions=4))
    save(OUT/'parent_heads.json',parents);save(OUT/'jobs.json',sorted(jobs,key=lambda j:j['job']));files += [OUT/'parent_heads.json',OUT/'jobs.json']
    finish(run,files,old_evidence=evidence,new_fits_performed=0,new_predictions_performed=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence),new_fits=0,new_predictions=0)),flush=True)

if __name__=='__main__':main()
