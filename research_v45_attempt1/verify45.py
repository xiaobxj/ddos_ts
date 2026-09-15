from common45 import *
from scipy.special import expit
from scipy.optimize import brentq
import math

def eq(a,b):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(float(a)-float(b))<2e-12,(a,b)
def independent_group(state,family):
    if family=='trend':return 'negative' if state in ['negative_low','negative_high'] else 'nonnegative'
    return 'low' if state in ['negative_low','nonnegative_low'] else 'high'
def same_tables(a,b):
    if len(a)==len(b)==0:return
    a=a.reset_index(drop=True).copy();b=b.reset_index(drop=True).copy()
    for col in a.columns:
        if a[col].isna().all() and b[col].isna().all():a[col]=np.nan;b[col]=np.nan
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=2e-12)
def independent_metric(g):
    n=len(g);y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool);q=g.probability.to_numpy(float);score=g.score.to_numpy(float);positive=int(y.sum());negative=n-positive
    r=dict(n=n,correct_directions=int((y==up).sum()),tp=int((y&up).sum()),tn=int((~y&~up).sum()),fp=int((~y&up).sum()),fn=int((y&~up).sum()))
    r.update(accuracy=r['correct_directions']/n if n else None,direction_error=(n-r['correct_directions'])/n if n else None,balanced_accuracy=(r['tp']/positive+r['tn']/negative)/2 if positive and negative else None,predicted_up_fraction=int(up.sum())/n if n else None,observed_up_fraction=positive/n if n else None,auroc=None,brier=None,log_loss=None,mean_probability=None,probability_std=None,calibration_gap=None,clipped_probabilities=None)
    if positive and negative:r['auroc']=math.fsum(float(a>b)+.5*float(a==b) for a in score[y] for b in score[~y])/(positive*negative)
    if n and np.isfinite(q).all():
        mean=math.fsum(q)/n;bounded=np.clip(q,1e-12,1-1e-12);r.update(brier=math.fsum((float(p)-int(v))**2 for p,v in zip(q,y))/n,log_loss=-math.fsum(math.log(float(p)) if v else math.log1p(-float(p)) for p,v in zip(bounded,y))/n,mean_probability=mean,probability_std=math.sqrt(math.fsum((float(p)-mean)**2 for p in q)/n),calibration_gap=mean-positive/n,clipped_probabilities=int(((q<1e-12)|(q>1-1e-12)).sum()))
    return r
def check_fits_and_validation():
    bank=csv('weekly_signal_bank');members=csv('split_membership');schedule=csv('schedule');heads=read(OUT/'correction_heads.json');gates=csv('gate_decisions');vp=csv('validation_predictions');bi=bank.set_index(['method','seed','row_index']);rows={k:g.row_index.tolist() for k,g in members.groupby(['cutoff','role'])};hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};state=bank[['row_index','state']].drop_duplicates().set_index('row_index').state
    assert len(heads)==1104 and len(hi)==1104 and len(gates)==368
    fitcount=0;maxgap=0.;maxkkt=0.;records=[]
    for h in heads:
        ids=[i for i in rows.get((h['cutoff'],'training'),[]) if independent_group(state[i],h['family'])==h['component']];z=[float(bi.loc[(h['method'],h['seed'],i),'annual_logit']) for i in ids];y=[int(bi.loc[(h['method'],h['seed'],i),'actual_up']) for i in ids]
        assert h['training_n']==len(ids) and h['training_up_n']==sum(y) and h['fit_eligible']==(h['mode']=='ready' and len(ids)>=10)
        if not h['fit_eligible']:assert h['offset']==0. and h['objective'] is None;continue
        def gradient(d):return math.fsum(float(expit(a+d))-b for a,b in zip(z,y))+20*d
        lo,high=gradient(-.5),gradient(.5)
        alt=-.5 if lo>=0 else .5 if high<=0 else brentq(gradient,-.5,.5,xtol=1e-14,rtol=1e-14)
        delta=h['offset'];gap=abs(alt-delta);assert gap<1e-11;maxgap=max(maxgap,gap)
        value=math.fsum(float(np.logaddexp(0.,-(a+delta) if b else a+delta)) for a,b in zip(z,y))+10*delta*delta;assert abs(value-h['objective'])<1e-10
        grad=gradient(delta);violation=max(0,-grad) if delta==-.5 else max(0,grad) if delta==.5 else abs(grad);assert violation<1e-10;maxkkt=max(maxkkt,violation)
        assert value<=math.fsum(float(np.logaddexp(0.,-a if b else a)) for a,b in zip(z,y))+1e-10
        records.append(dict(cutoff=h['cutoff'],method=h['method'],seed=h['seed'],family=h['family'],component=h['component'],coefficient_gap=gap,kkt_violation=violation));fitcount+=1
    same_tables(pd.DataFrame(heads),csv('correction_parameters'))
    vpi=vp.set_index(['cutoff','method','seed','family','component','row_index']);expected_vp=set()
    for r in gates.itertuples():
        ids=[i for i in rows.get((r.cutoff,'validation'),[]) if independent_group(state[i],r.family)==r.component];bases=[];probs=[];labels=[]
        for idx in ids:
            bp=[];pq=[]
            for seed in cfg()['seeds']:
                key=(r.cutoff,r.method,seed,r.family,r.component,idx);expected_vp.add(key);x=vpi.loc[key];b=bi.loc[(r.method,seed,idx)];h=hi[(r.cutoff,r.method,seed,r.family,r.component)]
                expected=b.probability if h['offset']==0 else float(expit(b.annual_logit+h['offset']));eq(x.candidate_probability,expected);assert x.offset==h['offset'] and x.annual_probability==b.probability and x.actual_up==b.actual_up and x.joint_completed==b.joint_completed<=r.cutoff and x.state==b.state
                bp.append(x.annual_probability);pq.append(x.candidate_probability)
            bases.append(math.fsum(bp)/3);probs.append(math.fsum(pq)/3);labels.append(int(b.actual_up))
        n=len(ids);bd=math.fsum((p-y)**2-(b-y)**2 for p,b,y in zip(probs,bases,labels))/n if n else None;bc=sum((p>.5)==y for p,y in zip(bases,labels));cc=sum((p>.5)==y for p,y in zip(probs,labels));eq(r.brier_difference,bd)
        reason=r.mode if r.mode!='ready' else 'insufficient_state_train' if r.training_n<10 else 'insufficient_validation' if n<5 else 'brier_not_better' if bd>=-1e-12 else 'direction_worse' if cc<bc else 'accepted'
        assert r.reason==reason and r.accepted==(reason=='accepted') and [r.validation_n,r.baseline_correct,r.candidate_correct]==[n,bc,cc]
    assert len(vp)==len(expected_vp) and set(vpi.index)==expected_vp
    fitpoison=0;valpoison=0
    for c in schedule.itertuples():
        sub=schedule[schedule.cutoff.eq(c.cutoff)];ids=rows.get((c.cutoff,'training'),[]);poison=bank.copy();mask=~poison.row_index.isin(ids);poison.loc[mask,'annual_logit']=99.;poison.loc[mask,'actual_up']=1-poison.loc[mask,'actual_up']
        alt=fit_records(poison,members,sub);expected=[h for h in heads if h['cutoff']==c.cutoff];assert alt==expected;fitpoison+=1
        poison=bank.copy();mask=poison.joint_completed.gt(c.cutoff);poison.loc[mask,'actual_up']=1-poison.loc[mask,'actual_up'];poison.loc[mask,'annual_logit']=-99.;poison.loc[mask,'probability']=.999
        gg,vv=validate_records(poison,members,sub,heads);same_tables(gg,gates[gates.cutoff.eq(c.cutoff)]);same_tables(vv,vp[vp.cutoff.eq(c.cutoff)]);valpoison+=1
    p=OUT/'independent_scalar_checks.csv';pd.DataFrame(records).to_csv(p,index=False)
    return dict(independent_scalar_solutions=fitcount,maximum_coefficient_gap=maxgap,maximum_kkt_violation=maxkkt,independent_gate_cells=368,independent_validation_seed_rows=len(vp),outside_training_poison_cutoffs=fitpoison,postcutoff_validation_poison_cutoffs=valpoison),[p]
def check_predictions_metrics():
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');details=csv('seed_routing');bank=csv('weekly_signal_bank');heads=read(OUT/'correction_heads.json');gates=csv('gate_decisions').set_index(['cutoff','method','family','component']);hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};bi=bank.set_index(['method','seed','row_index']);mi=models.set_index(['history','method','seed','row_index']);ei=ensemble.set_index(['history','method','row_index']);route=csv('routing').set_index('row_index')
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,check_exact=True)
    assert len(models)==108800 and len(ensemble)==40800 and len(details)==6528 and not details[['history','method','seed','row_index']].duplicated().any();maxgap=0.;zeros=0
    for r in details.itertuples():
        b=bi.loc[(r.method,r.seed,r.row_index)];group=independent_group(b.state,r.family);assert r.component==group and POLICIES[r.family]==r.history and r.cutoff==route.loc[r.row_index,'head_cutoff']<r.date
        h=hi[(r.cutoff,r.method,r.seed,r.family,group)];gate0=gates.loc[(r.cutoff,r.method,r.family,group)];off=h['offset'] if gate0.accepted else 0.;a=mi.loc[(r.history,r.method,r.seed,r.row_index)]
        p=b.probability if off==0 else float(expit(float(b.annual_logit)+off));gap=abs(a.probability-p);assert gap<1e-12;maxgap=max(maxgap,gap)
        assert r.offset==off and r.validation_accepted==gate0.accepted and r.fit_eligible==h['fit_eligible'] and r.gate_reason==gate0.reason and r.state==b.state and r.probability==a.probability and r.annual_probability==b.probability
        assert a.score==a.probability and a.direction_up==int(a.probability>.5) and a.actual_up==b.actual_up and a.actual==b.actual and a.joint_completed==b.joint_completed and a.cutoff==r.cutoff
        if off==0:assert a.probability==b.probability;zeros+=1
        if int(r.date[5:7])<=3:assert off==0
    for history in NEW:
        a=models[models.history.eq(history)&~models.method.isin(METHODS)].drop(columns='history').reset_index(drop=True);b=base[base.history.eq(ANNUAL)&~base.method.isin(METHODS)].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
    for (history,method,idx),g in models.groupby(['history','method','row_index'],sort=False):
        n=1 if method=='training_frequency' else 3;assert len(g)==n;r=ei.loc[(history,method,idx)];eq(r.score,math.fsum(g.score)/n)
        if method=='native_mse':assert pd.isna(r.probability) and r.direction_up==int(r.score>0)
        else:eq(r.probability,math.fsum(g.probability)/n);assert r.direction_up==int(r.probability>.5)
    joined=ensemble.merge(csv('weekly_context')[['row_index','date','state']],on=['row_index','date'],validate='many_to_one');count=0
    for name,source,keys in [('ensemble_metrics',ensemble,['history','method']),('seed_metrics',models,['history','method','seed']),('state_metrics',joined,['history','method','state'])]:
        groups={k:g for k,g in source.groupby(keys,sort=False)}
        for row in csv(name).to_dict('records'):
            w=next(w for w in periods() if w['name']==row['period']);g=groups.get(tuple(row[k] for k in keys),source.iloc[:0]);g=g[g.date.between(w['start'],w['end'])]
            for k,x in independent_metric(g).items():eq(x,row[k])
            count+=1
    return dict(independent_learned_seed_predictions=len(details),exact_zero_offset_predictions=zeros,maximum_prediction_gap=maxgap,independent_metric_cells=count)
def check_diagnostics():
    weekly=csv('weekly_policy_effects');e=csv('ensemble_predictions').set_index(['history','method','row_index']);assert len(weekly)==2176
    for r in weekly.itertuples():
        a=e.loc[(r.history,r.method,r.row_index)];assert r.probability==a.probability and r.actual_up==a.actual_up
        for label,history in [('annual',ANNUAL),('quarter4',QUARTER)]:
            b=e.loc[(history,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;expected='recovery' if good and not old else 'regression' if old and not good else 'stable_correct' if good else 'stable_wrong'
            assert getattr(r,label+'_probability')==b.probability and getattr(r,'case_vs_'+label)==expected and getattr(r,'changed_vs_'+label)==(a.probability!=b.probability);eq(getattr(r,'brier_vs_'+label),(a.probability-a.actual_up)**2-(b.probability-b.actual_up)**2)
    for name in ['policy_coverage','reference_comparisons']:
        for r in csv(name).itertuples():
            w=next(w for w in periods() if w['name']==r.period);g=weekly[weekly.history.eq(r.history)&weekly.method.eq(r.method)&weekly.date.between(w['start'],w['end'])];assert r.n==len(g)
            if name=='policy_coverage':assert [r.fit_eligible_weeks,r.validation_accepted_weeks,r.changed_probability_weeks]==[sum(g.fit_eligible),sum(g.validation_accepted),sum(g.changed_vs_annual)]
            else:
                label='annual' if r.reference_history==ANNUAL else 'quarter4';assert [r.changed_probability_weeks,r.recoveries,r.regressions]==[sum(g['changed_vs_'+label]),sum(g['case_vs_'+label]=='recovery'),sum(g['case_vs_'+label]=='regression')];eq(r.brier_difference,math.fsum(g['brier_vs_'+label])/len(g))
    for name in ['decision_outcomes','substate_outcomes']:
        for r in csv(name).itertuples():
            g=weekly[weekly.history.eq(r.history)&weekly.method.eq(r.method)&weekly.cutoff.eq(r.cutoff)];g=g[g.component.eq(r.component)] if name=='decision_outcomes' else g[g.state.eq(r.state)]
            assert [r.n,r.recoveries,r.regressions]==[len(g),sum(g.case_vs_annual=='recovery'),sum(g.case_vs_annual=='regression')];eq(r.brier_difference,math.fsum(g.brier_vs_annual)/len(g) if len(g) else None)
    support=csv('support_comparison');assert len(support)==184
    for r in csv('support_summary').itertuples():
        g=support[support.family.eq(r.family)];z=g[g['mode'].eq('ready')];assert [r.total_cells,r.ready_cells,r.training_supported_cells,r.validation_supported_cells]==[len(g),len(z),sum(z.train_supported),sum(z.validation_supported)];eq(r.training_support_fraction,sum(z.train_supported)/len(z));eq(r.validation_support_fraction,sum(z.validation_supported)/len(z))
    old=csv('original_four_state_support').rename(columns={'state':'component'});old['family']='four_state';same_tables(support[support.family.eq('four_state')],old[support.columns]);same_tables(support[support.family.ne('four_state')],csv('group_support'));same_tables(csv('all_2026_cases'),weekly[weekly.year.eq(2026)])
    return dict(independent_weekly_effects=len(weekly),independent_decision_outcomes=len(csv('decision_outcomes')),independent_substate_outcomes=len(csv('substate_outcomes')))
def check_statistics():
    e=csv('ensemble_predictions');pairs=read(OUT/'primary_comparisons.json');assert len(pairs)==48;expected_specs=[dict(window=w['name'],**s) for w in cfg()['windows'] for s in cfg()['primary_comparisons_per_window']];assert [{k:r[k] for k in spec} for r,spec in zip(pairs,expected_specs)]==expected_specs;ps=[]
    for w in cfg()['windows']:
        n=w['n'];starts=np.random.default_rng(20260910).integers(n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);p=e[e.date.between(w['start'],w['end'])]
        for r in [x for x in pairs if x['window']==w['name']]:
            a=p[p.history.eq(r['history'])&p.method.eq(r['candidate'])].sort_values('date');b=p[p.history.eq(r['reference_history'])&p.method.eq(r['reference'])].sort_values('date');assert len(a)==len(b)==n and a.date.tolist()==b.date.tolist()
            delta=a.direction_up.ne(a.actual_up).to_numpy(float)-b.direction_up.ne(b.actual_up).to_numpy(float) if r['metric']=='direction_error' else (a.probability.to_numpy()-a.actual_up.to_numpy())**2-(b.probability.to_numpy()-b.actual_up.to_numpy())**2
            mean=float(delta.mean());boots=delta[ids].mean(1);center=(delta-mean)[ids].mean(1);pvalue=(1+sum(abs(center)>=abs(mean)))/10001;lo,hi=np.quantile(boots,[.025,.975]);eq(mean,r['difference']);eq(lo,r['ci95_low']);eq(hi,r['ci95_high']);assert pvalue==r['p'];ps.append(pvalue)
    adjusted=[0.]*48;last=0.
    for rank,i in enumerate(sorted(range(48),key=lambda i:ps[i])):last=max(last,min(1.,ps[i]*(48-rank)));adjusted[i]=last
    assert adjusted==[r['holm_adjusted_p'] for r in pairs]
def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();start=now();last=read(OUT/'contract_verification.json')['completed_utc']
    for phase in ['fitting','validation','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    fit,files=check_fits_and_validation();print('Independent scalar optima, gates and training/validation information isolation PASS.',flush=True)
    pred=check_predictions_metrics();diag=check_diagnostics();check_statistics();assert old_evidence()==prep['old_evidence'];check_frozen()
    save(OUT/'verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5848,old_histories_preserved=23,new_neural_fits=0,independent_comparisons=48,**fit,**pred,**diag,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('R45 independent predictions, metrics, diagnostics and48comparisons PASS.',flush=True)
if __name__=='__main__':main()
