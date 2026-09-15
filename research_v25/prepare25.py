from common25 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists();old=previous.cfg();pairs=[]
    for method,parent in CANDIDATES.items():
        for ref,measures in [(parent,['brier','direction_error']),(WEAK[method],['brier','direction_error']),(OLD_INTERACTIONS[method],['brier','direction_error']),('training_frequency',['brier']),('order_only',['brier'])]:pairs.extend(dict(candidate=method,reference=ref,metric=key) for key in measures)
    assert len(pairs)==16
    variants=[dict(method=m,parent=parent,source_method=WEAK[m],dimensions=31 if m.startswith('learned') else 28,free_parameters=1,free_intercepts=0,order_lambda=.01,fits=18 if m.startswith('learned') else 6) for m,parent in CANDIDATES.items()]
    p=dict(version=25,experiment='Fixed original volatility-interaction parent plus one fitted scalar sign-order correction',scope='Research only. Two families,one fixed lambda=.01. Lock all R19parent slopes AND intercept,fit only gamma on unchanged R23order feature. No penalty grid,blend,intercept recalibration,expert selector or new data. Compare with same-lambda jointly-refitted R23and parentR19; retain all33older methods.',
        folds=old['folds'],windows=old['windows'],seeds=old['seeds'],data=old['data'],order_state=old['order_state'],variants=variants,
        model='z_offset=z_R19+gamma*h_R23order. Parent includes originalvolinteraction. Parent coefficients,intercept,allinputtransforms,ray and orderprojection are immutable. h uses unchanged R23training normalization; no new normalization or projection. Setting gamma0 exactly recovers originalR19probabilities.',
        objective='J(gamma)=mean(binaryNLL(z_parent+gamma*h,y))+.01/2*gamma^2. Fixedparent penalty C=.01/2*sum(parent slopes^2) omitted during scalaroptimization but added for comparisons with R19andR23. At optimum J_fixed+C<=J_R19 and J_R23_joint<=J_fixed+C up to numerical tolerance. These are in-sample nesting properties,not prediction guarantees.',
        probe=dict(l2_lambda=.01,max_iterations=100,gradient_absolute_tolerance=1e-10,line_search_max_steps=40,armijo=.0001,objective_roundoff_allowance=1e-14,initial_gamma=0.,free_intercepts=0),
        independent_solver=dict(method='brentq root of scalargradient',xtol=1e-13,rtol=1e-14,max_iterations=200,bracket='[-B,+B], B=mean(abs(h))/.01+1 guarantees opposite gradient signs',coefficient_absolute_tolerance=1e-7,objective_absolute_tolerance=1e-10,training_probability_absolute_tolerance=1e-7,gradient_absolute_tolerance=1e-9,purpose='24independent bracketed roots with scipy expit and separate loss formula; no alternate forecast used.'),
        sequencing='Freeze protocol/core/inputs before contract and fits. Complete all24scalar fits before any new nextyear scoring. Annual labels mature by cutoff. No full-pipelineOOFclaim: inherited neuralfeatures,ray,parent and projection trained in-sample within fold. No new neuralforward or training.',
        provenance=old['provenance'],primary_comparisons_per_window=pairs,inference='32fixed comparisons jointlyHolm,16perwindow. Candidate-reference losses,negativebetter. Circular8retained observations,10000replicates,seed20260910 separately perperiod; centeredtwo-sidedp,marginal95percentinterval. Not adjusted for repeated historicaldesign choices. No pooledprimary.',
        descriptive_screen=old['descriptive_screen'],retention_target=old['retention_target'],metrics=old['metrics'],pooled=old['pooled'],
        diagnostics='All26state cells,35methods,6years and seeds. Exact decomposition z_fixed-z_joint=(z_parent-z_joint_parent)+(gamma_fixed-gamma_joint)*h. Locking parents also changes optimalgamma,so compare as one coefficient-constraint intervention; do not interpret as separate causal market effects or score untrained counterfactual variants. Report full delta,each component,and probability changes.',
        budget='24new scalarfits,24new trainablecoefficients,726frozenparentcoefficients,750serializedcombinedcoordinates,37860cumulative trainingrows.24reused orderprojections,0newprojectionfits.1044newprobabilities+14616oldmodelrecords=15660;9135ensembles=35*261dates.1820state groups,2261allmetricgroups,340reliabilitybins,32contrasts.3585oldfilespreserved,including prior failed-attempt archive.0new neuralforward/training/HMMfits.',
        user_steering='Preserve original late58.16percent and R23early50.83percent partial improvement. Test fixedparent one-coefficient correction to separate this restriction from joint refitting; no automatic replacement or independent-confirmation claim.')
    save(ROOT/'protocol.json',p);evidence=old_evidence();run=manifest('preparation');files=[]
    for name in ['observation_table','classification_baselines','historical_date_usage','training_states','validation_states','hmm_signal_states','training_signal_states','validation_signal_states','validation_daily_states','validation_stability','state_correlations','temporal_provenance']:
        path=OUT/f'{name}.csv';shutil.copyfile(V24/'results'/f'{name}.csv',path);files.append(path)
    sources=read(V24/'results/source_heads.json');assert len(sources)==24;jobs=[]
    for method,parent in CANDIDATES.items():
        for source in sources:
            if source['method']==WEAK[method]:jobs.append(dict(job=f"{method}_{source['cutoff']}_{source['seed']}",method=method,source_job=source['job'],parent_job=source['source_job'],cutoff=source['cutoff'],seed=source['seed'],dimensions=source['dimensions'],free_parameters=1,free_intercepts=0,order_lambda=.01))
    assert len(jobs)==24
    for name,value in [('source_heads',sources),('jobs',jobs)]:path=OUT/f'{name}.json';save(path,value);files.append(path)
    finish(run,files,old_evidence=evidence,new_candidate_fits=0,new_predictions=0,independent_holdout_dates=0);print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence))),flush=True)
if __name__=='__main__':main()
