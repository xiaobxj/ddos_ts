from common20 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (ROOT/'protocol.json').exists() and not (OUT/'preparation_manifest.json').exists(),'Preserve frozen attempt'
    old=previous.cfg();obs,price,returns=data();folds=[]
    for f in old['folds']:
        fold=dict(f,recent_start=f"{int(f['cutoff'][:4])-2}-01-01");tr,te=indices(obs,fold);pos=recent_positions(obs,tr,fold)
        fold['recent_train_n']=len(pos);folds.append(fold)
    assert [f['recent_train_n'] for f in folds]==[720,590,596,595,725,725]
    comparisons=[]
    for family,(fa,fi,ra,ri) in FACTORIAL.items():
        for candidate,reference,metric_name in [(ra,fa,'brier'),(ra,fa,'direction_error'),(ri,fi,'brier'),(ri,fi,'direction_error'),
            (ri,ra,'brier'),(ri,ra,'direction_error'),(ri,'training_frequency','brier'),(ri,'recent_frequency','brier')]:
            comparisons.append(dict(family=family,candidate=candidate,reference=reference,metric=metric_name))
    independent=dict(old['independent_solver']);independent['purpose']='Verify all48new classifiers; alternate coefficients never used for forecasts.'
    p=dict(version=20,experiment='Three-calendar-year final classifier window, keeping the existing interaction as a factorial comparator',
        user_steering='Retain round19volatility interaction and its later-period directional improvement as a continuing research hypothesis. Failure of cross-period screening does not erase the historical comparison or preclude testing it again in a controlled new setting.',
        scope='Research only. Same261historically examined dates. Recent means the final3calendar years before each historical cutoff, not new2026observations. Freeze full-past backbone, preprocessing and interaction basis; change only classifier loss sample membership. This is not an end-to-end recent-data model.',
        folds=folds,windows=old['windows'],seeds=old['seeds'],data=old['data'],variants=SPECS,
        window='Retain full-fold rows with signal_date>=January1(cutoff_year-2), already joint_completed<=cutoff. Exactly3calendar years, no window grid, recency weights, label-based exclusions or tuning. Earlier historical bars may form retained signals; only label fitting membership changes.',
        coordinates='Reuse unchanged standardized additive X from round18and augmented[X,h] from round19. Do not refit clipping quantiles, means,SDs,neural weights,signal ray,OLS projection or residual scale. The interaction ray still uses full-fold supervised parent coefficients; this inherited older-label path is intentional and is not OOF. This isolates fitting-window effects conditional on fixed coordinates; it does not remove all older-data influence.',
        interaction=old['interaction'],interaction_note='All round19transform parameters are inherited as-is; no new projection is fit. Recent-subset h need not have mean0,SD1 or remain orthogonal to X. Compare nested recent additive/interaction classifiers on identical rows and unchanged lambda; zero interaction slope still nests the recent additive head.',
        probe=old['probe'],independent_solver=independent,
        budget='48new classifier fits:18learned and6raw for each of additive and interaction.31608feature rows across fits. No new neural forwards/training/projection fitting.2088new per-model probabilities; reuse5220old model rows for7308total. Add261deterministic recent-frequency forecasts only to ensembles, like prior frequency baselines.17methods*261dates=4437ensemble records.1428fitted coefficients including intercepts. Preserve2958old evidence files.',
        forecasts='All48fits must finish before new held-out scoring. Three seed probabilities averaged for learned models; raw deterministic. Strict p>0.5. Retain all12old methods including round19interaction. Add recent_frequency per cutoff from the exact recent training labels. Keep full training_frequency too; metadata training_frequency continues to mean the old full-fold baseline.',
        primary_comparisons_per_window=comparisons,
        inference='32fixed contrasts:8per family per period. Candidate-reference loss,negativebetter. Circular8retained-observation blocks,10000replicates,seed20260910 separately each period. Centered two-sided p,Holm across32. Intervals not simultaneous and not adjusted for repeated historical design selection. No pooled primary test.',
        factorial_diagnostic='For each family/window and Brier/direction_error, report (recent_interaction-recent_additive)-(full_interaction-full_additive). Negative means the interaction loss increment is more favorable under the recent fitting window. Algebraic paired-loss diagnostic only, not a causal mechanism or extra hypothesis test.',
        descriptive_screen='Focus on recent interaction candidates.12strict flags per window: accuracy exceeds full interaction,recent additive,nativeMSE,full frequency,recent frequency(5); Brier below full interaction,recent additive,full frequency,recent frequency(4); logloss below recent frequency(1); at least2years accuracy exceeds nativeMSE and at least2years Brier below recent frequency(2). Ties fail. Bothwindows all12required for descriptive PASS. Additive heads are factorial controls; no posthoc winner promotion.',
        metrics=old['metrics'],pooled=old['pooled'],diagnostics='All annual,seed,state,reliability results. Use existing full-fold state definitions for consistent groups; no recent-state relabeling. Report direction changes, coefficient deltas and the inherited h distribution on full/recent/test inputs. Allstatecells<20marked sparse.',
        verification='Hash freeze of old evidence/core/protocol/inputs. Independently reconstruct date/maturity masks, original validation forecasts and exact fixed input joins. Verify nested recent models,Newton replay,L-BFGS-B independent objectives,all new/reused predictions,ensembles,624metric groups,160bins,32block comparisons and8factorial rows. No new neural replay or projection-fitting claim.')
    save(ROOT/'protocol.json',p);evidence=old_evidence();assert read(V19/'results/verification.json')['status']=='PASS'
    run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv','training_states.csv','validation_states.csv']:
        target=OUT/name;shutil.copyfile(V19/'results'/name,target);files.append(target)
    membership=[];summaries=[]
    for fold in folds:
        tr,te=indices(obs,fold);pos=recent_positions(obs,tr,fold);rows=tr[pos]
        for i in tr:membership.append(dict(cutoff=fold['cutoff'],row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],included_recent=bool(i in rows)))
        summaries.append(dict(cutoff=fold['cutoff'],recent_start=fold['recent_start'],full_train_n=len(tr),recent_train_n=len(rows),removed_n=len(tr)-len(rows),test_n=len(te),
            first_recent_date=obs.date.iloc[rows[0]],last_recent_date=obs.date.iloc[rows[-1]],last_recent_label_completed=obs.joint_completed.iloc[rows].max(),
            full_frequency=float((returns[tr]>0).mean()),recent_frequency=float((returns[rows]>0).mean())))
    for name,rows in [('window_membership',membership),('fold_summaries',summaries)]:
        path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    sources={};jobs=[]
    for spec in SPECS:
        for s in read(PROJECT/f"research_v{spec['source_round']}/results/heads.json"):
            if s['method']!=spec['source_method']:continue
            sources[s['job']]=s;job=dict(spec,source_job=s['job'],cutoff=s['cutoff'],seed=s['seed'],job=f"{spec['method']}_{s['cutoff']}_{s['seed']}")
            job['recent_parent_job']=f"{INTERACTIONS[spec['method']]}_{s['cutoff']}_{s['seed']}" if spec['interaction'] else None;jobs.append(job)
    assert len(jobs)==len(sources)==48
    save(OUT/'source_heads.json',list(sources.values()));save(OUT/'jobs.json',jobs);files += [OUT/'source_heads.json',OUT/'jobs.json']
    finish(run,files,old_evidence=evidence,new_primary_fits_performed=0,new_predictions_performed=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence),recent_train_counts=[f['recent_train_n'] for f in folds])),flush=True)

if __name__=='__main__':main()
