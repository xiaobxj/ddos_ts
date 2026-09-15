from common21 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (ROOT/'protocol.json').exists() and not (OUT/'preparation_manifest.json').exists(),'Preserve frozen attempt'
    old=previous.cfg();comparisons=[]
    for candidate,parent in CANDIDATES.items():
        for reference,metrics in [(parent,['brier','direction_error']),(OLD_INTERACTIONS[candidate],['brier','direction_error']),('training_frequency',['brier']),('hmm_only',['brier'])]:
            comparisons.extend(dict(candidate=candidate,reference=reference,metric=m) for m in metrics)
    comparisons.append(dict(candidate='hmm_only',reference='training_frequency',metric='brier'));assert len(comparisons)==13
    p=dict(version=21,experiment='Frozen annual two-state zero-mean Gaussian volatility HMM with one residualized soft-state by inherited-signal interaction',
        scope='Research only; same261previously evaluated dates. Retain all17round20methods including original round19interaction. Full history, equal row weights. No recency cutoff/decay, no neural training or forward calls, no strategy deployment.',
        folds=[{k:f[k] for k in ['cutoff','end','train_n','test_n']} for f in old['folds']],windows=old['windows'],seeds=old['seeds'],data=old['data'],
        hmm=dict(states=2,fixed_means=[0.,0.],initial_distribution=[.5,.5],variance_bounds=[1e-8,1.],transition_floor=1e-4,
            initial_variance_multipliers=[.5,2.],initial_stay_probability=.97,max_iterations=3000,gradient_infinity_tolerance=1e-7,
            likelihood_roundoff_tolerance=1e-8,minimum_variance_ratio=1.05,minimum_soft_occupancy=20),
        hmm_data='Daily close log returns on each trading bar from price anchor1 through last date<=cutoff. Valid only when both adjacent OHLC bars valid and closes finite positive. Invalid returns have neutral emissions but still advance a trading-bar transition. No filling, no compression of missing bars. Fixed initial distribution applied to first emission without a prior transition.',
        hmm_fit='Single deterministic constrained Baum-Welch EM initialization. Both emission means fixed zero, sort final states by variance. EM smoothed probabilities only for within-training likelihood estimation. Four free parameters per annual fit; local stationary point, not global optimum or observed ground-truth regimes.',
        hmm_features='At each classifier training signal use forward-filtered high-variance posterior from parameters fit on the whole pre-cutoff daily sequence. This training feature is in-sample, not historically prequential/OOF. At held-out dates all parameters frozen; continue from last training filtered posterior times transition, through every next-year bar. No held-out smoothing/refitting. All six HMMs and30heads finish before any new held-out filtering/scoring.',
        interaction=dict(old['interaction'],volatility='Replace standardized volatility20 by u=2*filtered_high_variance_probability-1. Refit only this new training OLS product projection and residual normalization. Reuse unchanged R18 additive X and supervised first25coefficient ray.'),
        control='hmm_only uses one coordinate u standardized by classifier-training mean and populationSD floor1e-6; same ridge logistic objective. Six deterministic fits. No learned signal inputs.',
        variants=[dict(method='learned_hmm_interaction',parent='learned_market',dimensions=30,fits=18),dict(method='raw_hmm_interaction',parent='raw_trend',dimensions=27,fits=6),dict(method='hmm_only',parent=None,dimensions=1,fits=6)],
        probe=old['probe'],independent_solver=old['independent_solver'],projection_verification=read(V19/'protocol.json')['projection_verification'],
        independent_hmm=dict(method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-10,maxiter=1000,maxls=40),objective_absolute_tolerance=1e-8,gradient_infinity_tolerance=1e-6,
            filtered_probability_absolute_tolerance=.001,log_domain_likelihood_tolerance=1e-8,log_domain_probability_tolerance=1e-8,log_domain_transition_count_tolerance=1e-6,
            purpose='Independent log-domain recursion, toy exhaustive state paths and finite differences; locally refine EM result as stationarity check only. Alternate parameters never generate delivered forecasts; no claim of global optimality.'),
        primary_comparisons_per_window=comparisons,
        inference='26fixed contrasts, jointly Holm adjusted. Candidate-reference Brier or direction error, negativebetter. Circular8retained-observation blocks,10000replicates,seed20260910 separately each period. Centered two-sided p. Intervals neither simultaneous nor adjusted for repeated historical design selection. No pooled primary.',
        descriptive_screen='11strict flags per candidate per period: accuracy above R18parent,R19interaction,nativeMSE,fullfrequency(4); Brier below R18parent,R19interaction,fullfrequency,hmm_only(4); logloss below fullfrequency(1); at least2years accuracy above nativeMSE and at least2years Brier below fullfrequency(2). Ties fail. Both periods all11for descriptive PASS. Control not eligible for automatic promotion.',
        diagnostics='Preserve old12state cells and add fixed HMM posterior buckets p<.2low,p>.8high,otherwiseuncertain; all cells including n<20. These thresholds only tabulate, never train/switch. All annual,seed,reliability,training fit and logit-component diagnostics. Probability is a latent-model posterior, not a known market label.',
        metrics=old['metrics'],pooled=old['pooled'],
        budget='6HMM fits on10924daily steps (10missing),24projections,30classifier fits on47325rows,738classifier coefficients plus24HMM free parameters.1305new model probabilities;7308old model records→8613;4437old ensemble records→5220(20methods*261).1462heldout daily filtering steps,2missing. Preserve3061old evidence files.852metricgroups,190reliabilitybins.',
        user_steering='Keep the original full-history interaction and its late2018-2020accuracy58.16percent; failure of a later variant does not erase that hypothesis. Identify a state before separate expert training; this bounded round adds only one conditional signal direction, not two large expert networks.',
        sources=['https://hmmlearn.readthedocs.io/en/stable/tutorial.html','https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html'])
    p['independent_solver']=dict(p['independent_solver'],purpose='Independently verify all30new convex heads; alternate coefficients never score held-out data.')
    save(ROOT/'protocol.json',p);evidence=old_evidence();run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv','training_states.csv','validation_states.csv']:
        path=OUT/name;shutil.copyfile(V20/'results'/name,path);files.append(path)
    obs,price,returns=data();daily=daily_returns(price);summaries=[]
    for f in p['folds']:
        tr,te=indices(obs,f);a,b=daily_extent(price,f)
        summaries.append(dict(cutoff=f['cutoff'],end=f['end'],train_n=len(tr),test_n=len(te),training_end_anchor=a,test_end_anchor=b,daily_training_steps=a,
            missing_training=int(np.isnan(daily[:a]).sum()),daily_test_steps=b-a,missing_test=int(np.isnan(daily[a:b]).sum()),last_train_date=price.date.iloc[a],last_test_date=price.date.iloc[b]))
    assert sum(r['daily_training_steps'] for r in summaries)==10924 and sum(r['daily_test_steps'] for r in summaries)==1462
    path=OUT/'fold_summaries.csv';pd.DataFrame(summaries).to_csv(path,index=False);files.append(path)
    sources=[];jobs=[]
    for candidate,parent in CANDIDATES.items():
        for s in read(V18/'results/heads.json'):
            if s['method']!=parent:continue
            sources.append(s);jobs.append(dict(method=candidate,source_job=s['job'],cutoff=s['cutoff'],seed=s['seed'],dimensions=s['dimensions']+1,job=f"{candidate}_{s['cutoff']}_{s['seed']}"))
    for f in p['folds']:jobs.append(dict(method='hmm_only',source_job=None,cutoff=f['cutoff'],seed=0,dimensions=1,job=f"hmm_only_{f['cutoff']}_0"))
    assert len(sources)==24 and len(jobs)==30
    for name,values in [('source_heads',sources),('jobs',jobs)]:path=OUT/f'{name}.json';save(path,values);files.append(path)
    finish(run,files,old_evidence=evidence,new_hmm_fits=0,new_classifier_fits=0,new_predictions=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence))),flush=True)

if __name__=='__main__':main()
