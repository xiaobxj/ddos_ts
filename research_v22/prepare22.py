from common22 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists();old=previous.cfg();pairs=[]
    for m,parent in CANDIDATES.items():
        for ref,ms in [(parent,['brier','direction_error']),(OLD_INTERACTIONS[m],['brier','direction_error']),('training_frequency',['brier']),(CONTROLS[GATE_MAP[m]],['brier'])]:pairs.extend(dict(candidate=m,reference=ref,metric=metric) for metric in ms)
    pairs.extend(dict(candidate=m,reference='training_frequency',metric='brier') for m in CONTROLS.values());assert len(pairs)==26
    variants=[dict(method=m,parent=parent,gate=GATE_MAP[m],dimensions=30 if m.startswith('learned') else 27,fits=18 if m.startswith('learned') else 6) for m,parent in CANDIDATES.items()]+[dict(method=m,parent=None,gate=g,dimensions=1,fits=6) for g,m in CONTROLS.items()]
    p=dict(version=22,experiment='Two separate fixed causal state gates: relative volatility and log-price path efficiency, with one shared signal modifier each',
        scope='Research only. Retain all20old methods and original round19interaction. Test the two gate definitions separately, no joint gate or new expert networks. Full-history equal-weight labels and unchanged R18feature coordinates and supervised ray.',
        folds=old['folds'],windows=old['windows'],seeds=old['seeds'],data=old['data'],variants=variants,
        states=dict(relative_volatility='sd20,sd120 are populationSD of trailing20and120valid daily close log returns. u=(sd20-sd120)/max(sd20+sd120,1e-8). Bounded[-1,1]; dimensionless change in scale relative to trailing baseline.',
            persistence='ER60=clip(abs(sum trailing60close log returns)/max(sum absolute trailing60log returns,1e-8),0,1); u=2*ER60-1. Uses absolute log-price displacement, so it measures path efficiency regardless of direction. A log-price adaptation of efficiency ratio, not a claim of the standard raw-price KER implementation.',
            flat_case='Both SDs zero gives relativevol0; zero path length gives ER0, persistence-1. Constants fixed; no horizon, threshold, smoothing or parameter search.',
            memory='Require all121OHLC bars through signal t valid,close finitepositive. First120anchors missing. A bad bar renders anchors bad_index through bad_index+120missing. No filling or reindex compression. Existing125bar observation mask already stricter; no changed classifier training/test membership.',
            causal='No fitted state parameters; identical past-only function for training and next-year dates. Features on any prefix invariant to later data; no state identifier OOF-fitting mismatch. The inherited backbone/ray, OLS projection and classifier still fit on each whole training fold, so the entire learner is not OOF.'),
        interaction=dict(old['interaction'],volatility='Replace HMM gate by exactly one relativevol or persistence coordinate. Same frozen parent,train-only OLS residualization,SDfloor1e-6,rcond1e-12 and one extra slope.'),
        controls='Two separate one-coordinate standardized-state logistic controls; six annual heads each, same lambda. No signal inputs.',
        probe=old['probe'],independent_solver=dict(old['independent_solver'],purpose='Independently verify all60convex heads; alternate coefficients never predict heldout dates.'),projection_verification=old['projection_verification'],
        stability='Descriptive only: each gate on pre-cutoff daily history and next-year daily bars; fixed labels relativevol>0elevated,otherwise subdued; ER>=.2persistent,otherwisechoppy. Fourcrosscells too. Report occupancy,mean,SD,10/90quantiles,adjacent daily autocorrelation,switches,runs. Missing bars break adjacency; annual/history boundaries censor runs. Observed run lengths including censored runs are not true market-state durations. Thresholds for tabulation only.',
        training_diagnostics='Full-history state features are deterministic and causal. Classifier training signals and states retained. Gate novelty diagnostic: correlation with existing raw volatility20 and absolute trend60 descriptors, plus training OLS R2 from existing standardized additive X. These are descriptive collinearity checks, not selection or causal identification.',
        primary_comparisons_per_window=pairs,inference='52fixed contrasts (26perwindow),jointHolm. Candidate-reference loss negativebetter; circular8retained-observation blocks,10000replicates,seed20260910 separately perwindow. Centered two-sided p; marginal95percentintervals not selection-adjusted. No pooled primary.',
        descriptive_screen='11strict flags per candidate per window: accuracy above R18parent,R19interaction,nativeMSE,fullfrequency(4); Brier below parent,originalinteraction,frequency,matchingstateonly(4); logloss belowfrequency(1); at least2years accuracy above native and2years Brier belowfrequency(2). Bothwindows all11for descriptive PASS. No automatic selection of best of two gates or promotion of controls.',
        metrics=old['metrics'],pooled=old['pooled'],budget='60heads,48projections,94650training feature rows,1476coefficients includingintercepts.2610new probabilities,8613old modelrows→11223;5220old ensembles→6786(26methods*261).Preserve3166old evidencefiles.1520metricgroups including23statecells*26methods*2windows;250reliabilitybins;52primarycontrasts. No new neural forward/training/HMMfit.',
        sequencing='Freeze protocol/core/inputs before real candidate fitting. Contract synthetic/causality checks then all60heads finish before new heldout scoring. Next-year raw state diagnostics computed during scoring. Some earlier evaluation dates legitimately belong to later mature-label training folds.',
        user_steering='Preserve original full-history interaction including lateaccuracy58.16percent; inspect stable meaning and incremental content before training larger separate experts. A failed new variant does not erase prior results.',
        sources=['https://doc.stocksharp.com/en/topics/api/indicators/list_of_indicators/kaufman_efficiency_ratio','https://scikit-learn.org/stable/modules/cross_validation.html'])
    save(ROOT/'protocol.json',p);evidence=old_evidence();run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv','training_states.csv','validation_states.csv','hmm_signal_states.csv']:
        target=OUT/name;shutil.copyfile(V21/'results'/name,target);files.append(target)
    sources=[];jobs=[]
    originals=read(V18/'results/heads.json')
    for m,parent in CANDIDATES.items():
        for s in originals:
            if s['method']!=parent:continue
            if not any(x['job']==s['job'] for x in sources):sources.append(s)
            jobs.append(dict(method=m,source_job=s['job'],gate=GATE_MAP[m],cutoff=s['cutoff'],seed=s['seed'],dimensions=s['dimensions']+1,job=f"{m}_{s['cutoff']}_{s['seed']}"))
    for g,m in CONTROLS.items():
        for f in p['folds']:jobs.append(dict(method=m,source_job=None,gate=g,cutoff=f['cutoff'],seed=0,dimensions=1,job=f"{m}_{f['cutoff']}_0"))
    assert len(jobs)==60 and len(sources)==24
    for name,value in [('jobs',jobs),('source_heads',sources)]:path=OUT/f'{name}.json';save(path,value);files.append(path)
    finish(run,files,old_evidence=evidence,new_candidate_fits=0,new_predictions=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence))),flush=True)

if __name__=='__main__':main()
