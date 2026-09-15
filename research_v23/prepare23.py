from common23 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists();old=previous.cfg();pairs=[]
    for m,parent in CANDIDATES.items():
        for ref,metrics in [(parent,['brier','direction_error']),(OLD_INTERACTIONS[m],['brier','direction_error']),('training_frequency',['brier']),('order_only',['brier'])]:pairs.extend(dict(candidate=m,reference=ref,metric=k) for k in metrics)
    pairs.append(dict(candidate='order_only',reference='training_frequency',metric='brier'));assert len(pairs)==13
    variants=[dict(method=m,parent=parent,gate='excess_order',dimensions=31 if m.startswith('learned') else 28,fits=18 if m.startswith('learned') else 6) for m,parent in CANDIDATES.items()]+[dict(method='order_only',parent=None,gate='excess_order',dimensions=1,fits=6)]
    p=dict(version=23,experiment='Add one permutation-centered sign-order by signal interaction to the existing R19volatility-interaction model',
        scope='Research only. Retain all26old methods. One new deterministic60day sign-order gate, two feature-family extensions and one gate-only control. The parent now is R19originalvolinteraction, not R18additive; R22comparison is descriptive, not isolated causal attribution to newgate.',
        folds=old['folds'],windows=old['windows'],seeds=old['seeds'],data=old['data'],variants=variants,
        order_state=dict(signs='s_i=+1 if close_i>close_i-1, -1 if less,0 if exactly equal, on trailing60dailychanges. Use direct close comparison for neutral signs; no epsilon, demeaning or fitted classification threshold.',
            observed='A=(1/59)*sum(s_i*s_i-1), i=2..60. Positive adjacent product for same nonzero signs,negative for opposite,zero when either sign is0.',
            permutation_expectation='B=((sum s)^2-sum(s^2))/(60*59). Exact expected adjacentproduct when uniformly permuting the fixed60sign multiset. Not an estimate from future dates or random shuffles.',
            gate='u=A-B. Positive indicates excess same-sign adjacency conditional on sign counts,negative excess alternation. Unstandardized before product; no clipping or lag/horizon grid. A and B are each in[-1,1]; u is a difference,not a posterior probability or a standard correlation statistic.',
            limits='Allpositive,allnegative or allzero windows give u=0: with a single sign there is no order variation conditional on counts. Gate deliberately removes this count-imbalance explanation; it is not a universal trend detector. It ignores return magnitudes and is invariant to time reversal, though generally not arbitrary permutations.',
            context='Require same121validOHLCbar context as R22for comparable state diagnostics; direct order formula itself needs61closes. Existing125bar samplemask is stricter,so no changes to classifier membership. Bad bars break diagnosis; no filling or compression.',
            labels='For descriptivegroups only: u<-.05negative_order,u>.05positive_order,otherwise near_zero_order. No claim nearzero proves random order. Modelusescontinuousu.'),
        interaction=dict(parent='Reuse exactly the R19standardized[X18,h_original_vol] cache and next-year transform; retain old interaction coordinate. Add one new residualized product h_order to these coordinates.',
            ray='Reuse exactly R19cache.ray, originally R18parent.coefficients[:25]. Do not replace it by R19refitted coefficients. r=X_parent[:25]@ray, q=u*r. Same inherited supervised training ray is not OOF.',
            projection='Training-only OLS q~[X_parent,1], numpy lstsq rcond1e-12. Standardize residual by training mean/popSDfloor1e-6. Freeze ray/projection/scale for next-year application.',
            fit='Refit all parent slopes including originalvolinteraction,one addedorder slope,intercept. Same lambda=.01. Setting addedorder slope0 exactly nests frozen R19parent and its objective. Originalfeature retained; original coefficient is allowed to refit. Existing R19forecasts preserved separately.'),
        control='order_only: one training-standardizedu coordinate,popSDfloor1e-6,one slope/intercept,same loss andlambda,sixfits.',
        probe=old['probe'],independent_solver=dict(old['independent_solver'],purpose='Verify all30new convexheads; alternate coefficients never supply forecasts.'),projection_verification=old['projection_verification'],
        diagnostics='Training and next-year causal daily states,sign counts,observed/expected lagproduct,u,run statistics and censoring. Correlations with rawvol20,abs trend60,priorER60,relativevol. Training gate/product OLS R2. Separate refittedbase,originalinteraction and neworder logitcomponents. No subgroupselection or projection-based feature selection.',
        primary_comparisons_per_window=pairs,inference='26fixed contrasts jointlyHolm:13perperiod. Candidate-reference loss negativebetter. Circular8retained observations,10000replicates,seed20260910 separately perperiod; centeredtwo-sidedp and marginal95percentintervals. Not adjusted for repeated historicaldesign selection. No pooledprimary.',
        descriptive_screen='11strict flags per candidate/window: accuracy above R19parent,R18additive,nativeMSE,frequency(4); Brier below R19parent,R18additive,frequency,order_only(4); logloss belowfrequency(1); at least2yearsaccuracy>native and2yearBrier<frequency(2). Tiesfail; bothwindows all11required. Gate-onlycontrol not auto-promoted.',
        metrics=old['metrics'],pooled=old['pooled'],budget='30heads (18learned,6raw,6orderonly),24newprojections,47325trainingrows,762classifiercoefficients.1305newprobabilities,11223old modelrows→12528;6786oldensembles→7569=29methods*261.26statecells including23old+3new:1508stategroups;1868allmetricgroups;280reliabilitybins;26contrasts.3292oldfiles preserved. No neuralforward/training/HMMfit.',
        sequencing='Freeze protocol,core,inputs then contract; complete30fits before newheldoutscoring. Stateformulas are deterministic past-only,while parentfeatures,ray,projection andhead are training-insample. No new temporalOOF neuralrepresentation/ray claimed.',
        user_steering='Keep originalvolinteraction and its later58.16percent accuracy. Test genuinely order-sensitive content as an addition before largerexperts. Retain R22smallBrier improvement as a researchrecord.',
        sources=['https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm'],source_scope='Runs-based sequence randomness is background; this custom centered signproduct is not a NIST runs p-value/test or a validated marketstate label.')
    save(ROOT/'protocol.json',p);evidence=old_evidence();run=manifest('preparation');files=[]
    for name in ['observation_table.csv','classification_baselines.csv','historical_date_usage.csv','training_states.csv','validation_states.csv','hmm_signal_states.csv']:
        target=OUT/name;shutil.copyfile(V22/'results'/name,target);files.append(target)
    sources=[];jobs=[]
    for m,parent in CANDIDATES.items():
        for s in read(V19/'results/heads.json'):
            if s['method']!=parent:continue
            sources.append(s);jobs.append(dict(method=m,source_job=s['job'],gate='excess_order',cutoff=s['cutoff'],seed=s['seed'],dimensions=s['dimensions']+1,job=f"{m}_{s['cutoff']}_{s['seed']}"))
    for f in p['folds']:jobs.append(dict(method='order_only',source_job=None,gate='excess_order',cutoff=f['cutoff'],seed=0,dimensions=1,job=f"order_only_{f['cutoff']}_0"))
    assert len(sources)==24 and len(jobs)==30
    for name,value in [('source_heads',sources),('jobs',jobs)]:path=OUT/f'{name}.json';save(path,value);files.append(path)
    finish(run,files,old_evidence=evidence,new_candidate_fits=0,new_predictions=0,independent_holdout_dates=0);print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence))),flush=True)
if __name__=='__main__':main()
