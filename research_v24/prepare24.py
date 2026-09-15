from common24 import *

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists()
    old=previous.cfg();pairs=[]
    for method,parent in CANDIDATES.items():
        for ref,measures in [(parent,['brier','direction_error']),(WEAK[method],['brier','direction_error']),(OLD_INTERACTIONS[method],['brier','direction_error']),('training_frequency',['brier']),('order_only',['brier'])]:
            pairs.extend(dict(candidate=method,reference=ref,metric=k) for k in measures)
    assert len(pairs)==32
    variants=[dict(method=m,parent=CANDIDATES[m],source_method=WEAK[m],multiplier=f,order_lambda=.01*f,feature_scale=1/np.sqrt(f),dimensions=31 if m.startswith('learned') else 28,fits=18 if m.startswith('learned') else 6) for m,f in MULTIPLIERS.items()]
    p=dict(version=24,experiment='Two fixed levels of selective order-coefficient shrinkage; retain original volatility interaction',
        scope='Research only. Fixed4x and16x penalties for the one R23order slope. All other feature coordinates and penalties unchanged. Report both levels and both families; no best-level selection, adaptive gate, yearly switching or new expert model.',
        folds=old['folds'],windows=old['windows'],seeds=old['seeds'],data=old['data'],variants=variants,order_state=old['order_state'],
        objective='Mean binary NLL + .01/2 * sum(parent slopes squared) + lambda_order/2 * order_slope squared. Intercept unpenalized. lambda_order is .04 or .16; originalR23=.01 and parentR19has no order slope. Refit ALL coefficients including original volatility interaction. Setting neworder slope0 exactly nestsR19; lambda_order->infinity tends toR19. No intercept shift or postfit blend imposed.',
        implementation='Reuse unchanged R23train-standardized input. Divide only last orderfeature by sqrt(multiplier), then fit the existing isotropic .01 solver; do NOT re-standardize the scaled column. Map solver slope back by dividing it by sqrt(multiplier). Independently solve on unscaled R23input using explicit diagonal ridge penalty.',
        transform='Reuse all24R23order projections, rays, means and scales with hashes. No new projection or feature fitting. Original parent input and old volatility coordinate retained exactly. Never scale the old volatility coordinate.',
        probe=old['probe'],independent_solver=dict(old['independent_solver'],purpose='48independent L-BFGS-B solutions in original unscaled coordinates with diagonal penalty; no alternate forecast used.'),
        monotonicity='At fixed input and parent penalties, absolute optimal order coefficient must be nonincreasing as lambda_order rises .01,.04,.16. Verify this for all24sources. This does not imply any heldout metric or total-logit distance is monotone.',
        sequencing='Freeze all core code, protocol and inputs, then synthetic contract and provenance checks. Complete48newheads before any new nextyear forecast. Only annual training rows with matured labels used. No within-training OOF selection: cached learned representations, supervised ray, original interaction and residualization fitted on the full corresponding training fold. Do not call a last-head temporal split full-pipeline OOF.',
        provenance='Audit R23->R19->R18->R17->R16 feature-source job identities, cutoffs, exact row indices and label maturity. Reuse hash-preserved R17verification of18neural states; no new neural forward or checkpoint replay. Provenance is evidence from prior frozen records, not proof from a new full neural reconstruction.',
        primary_comparisons_per_window=pairs,inference='64fixed contrasts jointly Holm:32in each period. Candidate-reference losses, negative better. Circular8retained observations,10000replicates,seed20260910 independently by period; centeredtwo-sidedp,marginal95percentinterval. No correction for previous rounds of historical design adaptation; no pooledprimary.',
        descriptive_screen=old['descriptive_screen'],retention_target='Separately report whether each candidate has early accuracy>=R23weak extension, late accuracy>=R19parent, and Brier<=R23weak in BOTH periods. This is a frozen descriptive target, not statistical confirmation or promotion.',
        diagnostics='All inherited26state cells; all33methods,6years and seeds. Report original-coordinate order coefficients, parent coefficient drift, train and test order-logit SD and RMS probability change vsR19andR23. No state subgroup selection.',
        metrics=old['metrics'],pooled=old['pooled'],budget='48newheads,1500classifiercoefficients,75720trainingrows;0newprojections,24reusedprojections.2088newprobabilities +12528oldrecords=14616modelrecords.8613ensembles=33methods*261dates.1716state groups,2130allmetricgroups,320reliabilitybins,64contrasts;3389oldfiles preserved.0neural forward/training/HMMfits.',
        user_steering='Preserve R19late58.16percent and R23early50.83percent partial improvement. Test smaller order impact before any new learned state selector. No automatic replacement and no new independent dates.')
    # The generic solver config describes its scaled coordinate problem, not the effective order penalty.
    p['probe']=dict(old['probe'],objective='Isotropic .01 ridge in solver coordinates after scaling ONLY order input by1/sqrt(multiplier). Effective original-coordinate order penalty is .04/.16; intercept unpenalized.')
    p['attempt_history']='Attempt1 completed48heads and2088forecasts but evaluation stopped on a local-name collision. All85files and archive manifest preserved. Attempt2 fixes only the evaluator name and adds archival/replay verification; scientific design, samples,penalties and solver settings unchanged. Final48fits plus48archivedfits=96primary executions; no neural execution. Final2088new forecast records are recomputations, not additional unique dates.'
    p['budget']=p['budget'].replace('3389oldfiles preserved','3475evidencefiles preserved=3389priorround+86attempt1archive')
    save(ROOT/'protocol.json',p);evidence=old_evidence();run=manifest('preparation');files=[]
    for name in ['observation_table','classification_baselines','historical_date_usage','training_states','validation_states','hmm_signal_states','training_signal_states','validation_signal_states','validation_daily_states','validation_stability','state_correlations']:
        path=OUT/f'{name}.csv';shutil.copyfile(V23/'results'/f'{name}.csv',path);files.append(path)
    sources=[s for s in read(V23/'results/heads.json') if s['method'] in set(WEAK.values())];jobs=[];assert len(sources)==24
    for method,factor in MULTIPLIERS.items():
        for source in sources:
            if source['method']!=WEAK[method]:continue
            jobs.append(dict(job=f"{method}_{source['cutoff']}_{source['seed']}",method=method,source_job=source['job'],cutoff=source['cutoff'],seed=source['seed'],dimensions=source['dimensions'],multiplier=factor,order_lambda=.01*factor))
    assert len(jobs)==48
    for name,value in [('source_heads',sources),('jobs',jobs)]:path=OUT/f'{name}.json';save(path,value);files.append(path)
    finish(run,files,old_evidence=evidence,new_candidate_fits=0,new_predictions=0,independent_holdout_dates=0)
    print(json.dumps(dict(status='PREPARED',protocol_sha256=sha(ROOT/'protocol.json'),old_files=len(evidence))),flush=True)
if __name__=='__main__':main()
