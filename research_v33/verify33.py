from common33 import *
from contract33 import synthetic_cases,sources_check
from verify28 import independent_inputs
import math

def verify_windows(summary,members):
    obs,_,targets=data();old=read(OUT/'annual_heads.json');new=read(OUT/'quarter_heads.json')
    for r in summary.itertuples():
        c=pd.Timestamp(r.head_cutoff);e=pd.Timestamp(r.encoder_cutoff);a=set(np.flatnonzero(pd.to_datetime(obs.date).ge(e-pd.DateOffset(years=5)+pd.Timedelta(days=1))&pd.to_datetime(obs.joint_completed).le(e)));b=set(np.flatnonzero(pd.to_datetime(obs.date).ge(c-pd.DateOffset(years=5)+pd.Timedelta(days=1))&pd.to_datetime(obs.joint_completed).le(c)));assert len(a)==r.annual_train_n and len(b)==r.quarter_train_n
        for role,ids in [('retained',a&b),('added',b-a),('removed',a-b)]:
            g=members[members.head_cutoff.eq(r.head_cutoff)&members.role.eq(role)];assert g.row_index.tolist()==sorted(ids) and len(g)==getattr(r,role+'_n');assert g.date.tolist()==obs.date.iloc[g.row_index].tolist() and g.joint_completed.tolist()==obs.joint_completed.iloc[g.row_index].tolist();np.testing.assert_allclose(g.actual_return,targets['returns'][g.row_index],rtol=0,atol=1e-14)
            if ids:assert abs(sum(int(targets['returns'][i]>0) for i in ids)/len(ids)-getattr(r,role+'_up_fraction'))<1e-14
            else:assert pd.isna(getattr(r,role+'_up_fraction'))
        rate_a=sum(int(targets['returns'][i]>0) for i in a)/len(a);rate_b=sum(int(targets['returns'][i]>0) for i in b)/len(b);assert abs(rate_a-r.annual_up_fraction)<1e-14 and abs(rate_b-r.quarter_up_fraction)<1e-14 and abs(rate_b-rate_a-r.up_fraction_change)<1e-14
    assert len(summary)==23

def main():
    started=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();checks=sources_check();assert len(synthetic_cases())==9;run=check_phase('diagnosis');assert read(OUT/'contract_verification.json')['completed_utc']<=run['started_utc']<run['finished_utc'];component_manifest=read(OUT/'component_manifest.json');assert run['started_utc']<=component_manifest['completed_utc']<=run['finished_utc']
    for n,d in component_manifest['artifacts'].items():assert sha(ROOT/n)==d
    seed=csv('seed_components');lookup={(r.method,r.seed,r.date,r.group):r for r in seed.itertuples()};archive={(r.history,r.method,r.seed,r.date):r for r in csv('model_predictions').itertuples()};route_dates=csv('routing').set_index('row_index').date.to_dict();nodes,weights=np.polynomial.legendre.leggauss(64);nodes=(nodes+1)/2;weights=weights/2;maxterm=maxprob=maxweight=maxallocation=0.;count=0;parameter_rows=[];transform_rows=[]
    for job in jobs():
        xa=independent_inputs(job['features'],job['market'],job['gate'],job['old_data']);xn=independent_inputs(job['features'],job['market'],job['gate'],job['new_data']);pp,tt=parameter_changes(job);parameter_rows.extend(pp);transform_rows.extend(tt)
        for method,ha,hn,a,b in zip(METHODS,job['old_heads'],job['new_heads'],xa,xn):
            ta=np.asarray(ha['coefficients']);tn=np.asarray(hn['coefficients']);groups=[None,list(range(25)),list(range(25,29)),[29] if method!='learned_market' else [],[30] if method in ['learned_order_extension','learned_order_offset'] else []]
            for k,i in enumerate(job['rows']):
                date=route_dates[int(i)];oa=[];nb=[];co=[];tr=[]
                for group,indices in zip(GROUPS,groups):
                    if indices is None:av=float(ta[-1]);bv=float(tn[-1]);cv=bv-av;tv=0.
                    else:
                        av=math.fsum(float(a[k,j])*float(ta[j]) for j in indices);bv=math.fsum(float(b[k,j])*float(tn[j]) for j in indices);cv=math.fsum((float(tn[j])-float(ta[j]))*(float(b[k,j])+float(a[k,j]))/2 for j in indices);tv=math.fsum((float(tn[j])+float(ta[j]))*(float(b[k,j])-float(a[k,j]))/2 for j in indices)
                    oa.append(av);nb.append(bv);co.append(cv);tr.append(tv)
                za=math.fsum(oa);zn=math.fsum(nb);pa=float(expit(za));pn=float(expit(zn));q=expit(za+nodes*(zn-za));weight=float(np.dot(weights,q*(1-q)))
                for g,av,bv,cv,tv in zip(GROUPS,oa,nb,co,tr):
                    r=lookup[(method,job['seed'],date,g)];gap=max(abs(r.old_term-av),abs(r.new_term-bv),abs(r.coefficient_delta_logit-cv),abs(r.transform_delta_logit-tv));maxterm=max(maxterm,gap);assert gap<2e-12;assert abs(r.delta_logit-(bv-av))<2e-12 and abs(cv+tv-(bv-av))<2e-12
                    maxweight=max(maxweight,abs(weight-r.secant_weight));assert abs(weight-r.secant_weight)<2e-11
                    agap=max(abs(weight*(bv-av)-r.probability_contribution),abs(weight*cv-r.coefficient_probability_contribution),abs(weight*tv-r.transform_probability_contribution));maxallocation=max(maxallocation,agap);assert agap<2e-12;assert abs(r.coefficient_probability_contribution+r.transform_probability_contribution-r.probability_contribution)<2e-13
                    assert abs(r.old_logit-za)<2e-12 and abs(r.new_logit-zn)<2e-12
                old=archive[(cfg()['reference'],method,job['seed'],date)];new=archive[(cfg()['candidate'],method,job['seed'],date)];gap=max(abs(pa-old.probability),abs(pn-new.probability));maxprob=max(maxprob,gap);assert gap<2e-13 and int(pa>.5)==old.direction_up and int(pn>.5)==new.direction_up;count+=1
    assert count==3264 and len(seed)==16320
    pd.testing.assert_frame_equal(pd.DataFrame(parameter_rows),csv('parameter_changes'),check_dtype=False,atol=1e-13,rtol=0);pd.testing.assert_frame_equal(pd.DataFrame(transform_rows),csv('transform_changes'),check_dtype=False,atol=1e-13,rtol=0)
    weekly=csv('weekly_attribution');pd.testing.assert_frame_equal(weekly_from(seed),weekly,check_dtype=False,atol=1e-13,rtol=0);ensemble={(r.history,r.method,r.date):r for r in csv('ensemble_predictions').itertuples()};maxensemble=0.
    for r in weekly.itertuples():
        old=ensemble[(cfg()['reference'],r.method,r.date)];new=ensemble[(cfg()['candidate'],r.method,r.date)];pairs=[archive[(h,r.method,s,r.date)] for h in [cfg()['reference'],cfg()['candidate']] for s in cfg()['seeds']];pa=math.fsum(p.probability for p in pairs[:3])/3;pn=math.fsum(p.probability for p in pairs[3:])/3;assert abs(pa-r.old_probability)<1e-14 and abs(pn-r.new_probability)<1e-14
        contributions=[]
        for group in GROUPS:
            value=math.fsum(lookup[(r.method,s,r.date,group)].probability_contribution for s in cfg()['seeds'])/3;assert abs(value-getattr(r,'p_'+group))<1e-14;contributions.append(value)
        gap=abs(math.fsum(contributions)-(pn-pa));maxensemble=max(maxensemble,gap);assert gap<2e-13;assert r.old_correct==bool(old.direction_up==old.actual_up) and r.new_correct==bool(new.direction_up==new.actual_up)
        expected='regression' if r.old_correct and not r.new_correct else 'recovery' if r.new_correct and not r.old_correct else 'stable_correct' if r.old_correct else 'stable_wrong';assert r.case==expected
        signed=[(2*r.actual_up-1)*getattr(r,'p_'+group) for group in GROUPS];index=min(range(5),key=lambda j:signed[j]);assert r.dominant_adverse_group==(GROUPS[index] if signed[index]<-1e-12 else 'none')
        if r.head_cutoff==r.encoder_cutoff:assert r.probability_change==0 and all(getattr(r,prefix+group)==0 for prefix in ['p_','coef_','transform_'] for group in GROUPS)
    computed=summaries(weekly)
    for name,g in computed.items():pd.testing.assert_frame_equal(g,csv(name),check_dtype=False,atol=1e-13,rtol=0)
    for r in csv('component_summary').itertuples():
        g=weekly[weekly.method.eq(r.method)]
        if r.period.startswith('year_'):g=g[g.year.eq(int(r.period[-4:]))]
        elif r.period!='pooled_2021_2026':w=next(w for w in cfg()['windows'] if w['name']==r.period);g=g[g.date.between(w['start'],w['end'])]
        if r.case!='all':g=g[g.case.eq(r.case)]
        assert len(g)==r.n
        if r.n:
            signed=math.fsum((2*x.actual_up-1)*getattr(x,'p_'+r.group) for x in g.itertuples())/r.n;assert abs(signed-r.mean_signed_contribution)<1e-14
            for column,field in [('coef_','mean_signed_coefficient'),('transform_','mean_signed_transform')]:assert abs(math.fsum((2*x.actual_up-1)*getattr(x,column+r.group) for x in g.itertuples())/r.n-getattr(r,field))<1e-14
    for r in csv('performance_summary').itertuples():
        if r.period.startswith('year_'):
            g=csv('baseline_yearly_metrics');g=g[g.year.eq(int(r.period[-4:]))]
        else:g=csv('baseline_metrics');g=g[g.window.eq(r.period)]
        a=g[g.history.eq(cfg()['reference'])&g.method.eq(r.method)].iloc[0];b=g[g.history.eq(cfg()['candidate'])&g.method.eq(r.method)].iloc[0];assert r.n==a.n==b.n and r.annual_correct==a.correct_directions and r.quarter_head_correct==b.correct_directions and r.quarter_head_correct-r.annual_correct==r.recovery-r.regression
        assert abs(r.annual_brier-a.brier)<1e-14 and abs(r.quarter_head_brier-b.brier)<1e-14
    for name,g in [('all_2026_cases',weekly[weekly.year.eq(2026)]),('all_direction_flips',weekly[weekly.case.isin(['regression','recovery'])]),('flips_2026',weekly[weekly.year.eq(2026)&weekly.case.isin(['regression','recovery'])])]:pd.testing.assert_frame_equal(g.reset_index(drop=True),csv(name),check_dtype=False,atol=1e-14,rtol=0)
    shifts,members=window_changes();pd.testing.assert_frame_equal(shifts,csv('training_window_changes'),check_dtype=False,atol=1e-14,rtol=0);pd.testing.assert_frame_equal(members,csv('training_membership_changes'),check_dtype=False,atol=1e-14,rtol=0);verify_windows(shifts,members);assert old_evidence()==prep['old_evidence'];check_frozen()
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=4895,seed_week_checks=3264,component_checks=16320,ensemble_checks=1088,independent_transform_and_scalar_algebra=True,independent_64node_quadrature=True,all_q1_zero=True,training_windows_checked=23,maximum_term_gap=maxterm,maximum_archived_probability_gap=maxprob,maximum_quadrature_weight_gap=maxweight,maximum_probability_allocation_gap=maxallocation,maximum_ensemble_sum_gap=maxensemble,new_neural_fits=0,new_head_fits=0,new_policies=0,new_inferential_tests=0));print('Verification PASS: exact group accounting, nonlinear seed fusion, window memberships and old evidence.',flush=True)
if __name__=='__main__':main()
