from common40 import *
from contract40 import synthetic
from validate40 import validate
from verify37 import independent_metric,independently_solve
from scipy.special import expit
import math
def eq(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<tol,(a,b)
def check_splits():
    obs,_,targets=data();route=csv('routing');members=csv('split_membership');bank=csv('weekly_signal_bank');pool=obs.iloc[route.row_index].copy();pool['row_index']=route.row_index.to_numpy();source=csv('baseline_model_predictions');src=source[source.history.eq(ANNUAL)&source.method.isin(LEARNED)].set_index(['method','seed','row_index']);ctx=csv('weekly_context').set_index('row_index');nready=0
    for c in csv('schedule').itertuples():
        available=sorted([r for r in pool.itertuples() if isinstance(r.joint_completed,str) and r.joint_completed<=c.cutoff],key=lambda r:r.date);val=available[-13:];fit_cutoff=(pd.Timestamp(val[0].date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d') if val else None;train=sorted([r for r in pool.itertuples() if fit_cutoff and r.joint_completed<=fit_cutoff],key=lambda r:r.date)[-52:];mode='annual_reset' if c.cutoff.endswith('12-31') else 'q1_guard' if int(c.cutoff[5:7]) in (1,2) else 'ready' if len(train)==52 and len(val)==13 else 'insufficient_history';assert c.mode==mode and c.available==len(available) and c.train_n==len(train) and c.validation_n==len(val);eq(c.fit_cutoff,fit_cutoff);nready+=mode=='ready'
        for role,rows in [('training',train),('validation',val)]:
            g=members[members.cutoff.eq(c.cutoff)&members.role.eq(role)];assert g.row_index.tolist()==[r.row_index for r in rows] and g.used.eq(mode=='ready').all()
        if train:assert max(r.joint_completed for r in train)<min(r.date for r in val);assert max(r.exit for r in train)<=min(r.entry for r in val)
    for r in bank.itertuples():
        p=src.loc[(r.method,r.seed,r.row_index)];assert r.probability==p.probability and r.annual_probability==p.probability and r.actual_up==int(targets['returns'][r.row_index]>0);assert r.state==ctx.loc[r.row_index,'state'] and r.encoder_cutoff==ctx.loc[r.row_index,'encoder_cutoff']<r.date;assert abs(float(expit(r.annual_logit))-r.probability)<1e-12
    assert len(bank)==3264 and not bank.duplicated(['row_index','method','seed']).any();return nready
def check_fits_and_gates():
    bank=csv('weekly_signal_bank');members=csv('split_membership');heads=read(OUT/'correction_heads.json');schedule=csv('schedule');saved=csv('training_interface');sol=[];maxgap=0.;gate_checks=[];inputs=0;poisons=0
    hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};gates=csv('gate_decisions').set_index(['cutoff','method','family','component']);vp=csv('validation_predictions');vi=vp.set_index(['cutoff','method','seed','family','component','row_index'])
    for c in schedule.itertuples():
        trids=members[members.cutoff.eq(c.cutoff)&members.role.eq('training')].row_index.tolist();valids=members[members.cutoff.eq(c.cutoff)&members.role.eq('validation')].row_index.tolist()
        for m in LEARNED:
            for seed in cfg()['seeds']:
                tr=input_rows(bank,trids,m,seed);hs=[hi[(c.cutoff,m,seed,family,comp)] for family,comps in FAMILY_COMPONENTS for comp in comps]
                if c.mode=='ready':
                    table=saved[saved.cutoff.eq(c.cutoff)&saved.method.eq(m)&saved.seed.eq(seed)];assert table.row_index.tolist()==trids
                    for key in ['annual_logit','probability','actual_up']:np.testing.assert_array_equal(table[key],tr[key])
                    poisoned=bank.copy();outside=~poisoned.row_index.isin(trids);poisoned.loc[outside,'actual_up']=1-poisoned.loc[outside,'actual_up'];poisoned.loc[outside,['annual_logit','probability']]=1e8;assert fitted(input_rows(poisoned,trids,m,seed),c.mode)==[{k:v for k,v in h.items() if k not in ['cutoff','fit_cutoff','mode','method','seed']} for h in hs];inputs+=1
                for h in hs:
                    g=tr if h['component']=='all' else tr[tr.state.eq(h['component'])];eligible=c.mode=='ready' and (len(g)>=10 if h['family']=='state' else len(g)==52);assert h['n']==len(g) and h['fit_eligible']==eligible and h['up_n']==int(g.actual_up.sum())
                    if not eligible:assert h['offset']==0.;continue
                    d=independently_solve(g.annual_logit.to_numpy(),g.actual_up.to_numpy(),20.,.5);gap=abs(d-h['offset']);maxgap=max(maxgap,gap);assert gap<1e-10;assert h['kkt_violation']<1e-10;sol.append(dict(cutoff=c.cutoff,method=m,seed=seed,family=h['family'],component=h['component'],coefficient_gap=gap))
            for family,components in FAMILY_COMPONENTS:
                for component in components:
                    first=input_rows(bank,valids,m,cfg()['seeds'][0]);first=first if component=='all' else first[first.state.eq(component)];p0=[];p=[];y=first.actual_up.to_numpy(int);trn=hi[(c.cutoff,m,cfg()['seeds'][0],family,component)]['n']
                    for row in first.itertuples():
                        qs=[];bs=[]
                        for seed in cfg()['seeds']:
                            source=input_rows(bank,[row.row_index],m,seed).iloc[0];h=hi[(c.cutoff,m,seed,family,component)];q=float(source.probability) if h['offset']==0 else float(expit(source.annual_logit+h['offset']));qs.append(q);bs.append(float(source.probability));r=vi.loc[(c.cutoff,m,seed,family,component,row.row_index)];eq(r.candidate_probability,q);assert r.annual_probability==source.probability and r.offset==h['offset'] and r.actual_up==row.actual_up and r.joint_completed<=c.cutoff
                        p.append(math.fsum(qs)/3);p0.append(math.fsum(bs)/3)
                    p=np.array(p);p0=np.array(p0);n=len(y);diff=math.fsum((float(q)-int(v))**2-(float(b)-int(v))**2 for q,b,v in zip(p,p0,y))/n if n else None;bc=sum(int(b>.5)==int(v) for b,v in zip(p0,y));cc=sum(int(q>.5)==int(v) for q,v in zip(p,y));reason=c.mode if c.mode!='ready' else 'insufficient_state_train' if family=='state' and trn<10 else 'insufficient_validation' if n<(5 if family=='state' else 13) else 'brier_not_better' if not diff<-1e-12 else 'direction_worse' if cc<bc else 'accepted';r=gates.loc[(c.cutoff,m,family,component)];assert r.accepted==(reason=='accepted') and r.reason==reason and r.training_n==trn and r.validation_n==n and r.baseline_correct==bc and r.candidate_correct==cc;eq(r.brier_difference,diff);gate_checks.append(dict(cutoff=c.cutoff,method=m,family=family,component=component,reason=reason,accepted=reason=='accepted'))
        # With trained parameters sealed, only held-out rows can affect the gate.
        poisoned=bank.copy();outside=~poisoned.row_index.isin(valids);poisoned.loc[outside,'actual_up']=1-poisoned.loc[outside,'actual_up'];poisoned.loc[outside,['annual_logit','probability']]=1e8;a,b=validate(poisoned,heads,schedule[schedule.cutoff.eq(c.cutoff)],members);expected=csv('gate_decisions');expected=expected[expected.cutoff.eq(c.cutoff)].reset_index(drop=True);frames_equal_missing(a,expected,check_dtype=False,atol=1e-14,rtol=0);poisons+=1
    return pd.DataFrame(sol),pd.DataFrame(gate_checks),dict(independent_scalar_solutions=len(sol),maximum_scalar_gap=maxgap,training_isolation_interfaces=inputs,validation_isolation_cutoffs=poisons)
def check_predictions():
    bank=csv('weekly_signal_bank').set_index(['method','seed','row_index']);heads=read(OUT/'correction_heads.json');hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};gate=csv('gate_decisions').set_index(['cutoff','method','family','component']);models=csv('model_predictions');ensemble=csv('ensemble_predictions');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');routing=csv('seed_routing');mi=models.set_index(['history','method','seed','row_index']);maxgap=0.;fallback=0
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    inv={v:k for k,v in POLICIES.items()}
    for r in routing.itertuples():
        policy=inv[r.history];family=policy.split('_')[0];component=r.state if family=='state' else 'all';h=hi[(r.cutoff,r.method,r.seed,family,component)];g=gate.loc[(r.cutoff,r.method,family,component)];source=bank.loc[(r.method,r.seed,r.row_index)];off=h['offset'] if policy.endswith('direct') or g.accepted else 0.;p=float(source.probability) if off==0 else float(expit(source.annual_logit+off));actual=mi.loc[(r.history,r.method,r.seed,r.row_index)];gap=abs(actual.probability-p);maxgap=max(maxgap,gap);assert gap<1e-12
        assert r.offset==off and r.validation_accepted==g.accepted and r.fit_eligible==h['fit_eligible'] and r.gate_reason==g.reason and r.state==source.state and r.cutoff==monthly_for(r.date)<r.date;assert actual.direction_up==int(actual.probability>.5) and actual.actual_up==source.actual_up and r.probability==actual.probability
        if off==0:assert actual.probability==source.probability;fallback+=1
    assert len(routing)==6528
    for history in NEW:
        a=models[models.history.eq(history)&~models.method.isin(LEARNED)].drop(columns='history').reset_index(drop=True);b=base[base.history.eq(ANNUAL)&~base.method.isin(LEARNED)].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
    ei=ensemble.set_index(['history','method','row_index'])
    for (h,m,i),g in models.groupby(['history','method','row_index'],sort=False):
        n=1 if m=='training_frequency' else 3;r=ei.loc[(h,m,i)];assert len(g)==n;eq(r.score,math.fsum(g.score)/n)
        if m=='native_mse':assert pd.isna(r.probability) and r.direction_up==int(r.score>0)
        else:eq(r.probability,math.fsum(g.probability)/n);assert r.direction_up==int(r.probability>.5)
    count=0
    for name,source,keys in [('ensemble_metrics',ensemble,['history','method']),('seed_metrics',models,['history','method','seed']),('state_metrics',ensemble.merge(csv('weekly_context')[['row_index','date','state']],on=['row_index','date'],validate='many_to_one'),['history','method','state'])]:
        groups={k:g for k,g in source.groupby(keys,sort=False)}
        for r in csv(name).to_dict('records'):
            p=next(p for p in periods() if p['name']==r['period']);g=groups.get(tuple(r[k] for k in keys),source.iloc[:0]);g=g[g.date.between(p['start'],p['end'])]
            for k,v in independent_metric(g).items():eq(v,r[k])
            count+=1
    weekly=csv('weekly_policy_effects')
    for r in weekly.itertuples():
        a=ei.loc[(r.history,r.method,r.row_index)];b=ei.loc[(ANNUAL,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;case='regression' if old and not good else 'recovery' if good and not old else 'stable_correct' if old else 'stable_wrong';assert r.case==case and r.probability==a.probability and r.annual_probability==b.probability and r.changed_probability==(a.probability!=b.probability)
    for r in csv('direction_changes').itertuples():
        p=next(p for p in periods() if p['name']==r.period);g=weekly[weekly.history.eq(r.history)&weekly.method.eq(r.method)&weekly.date.between(p['start'],p['end'])];assert r.n==len(g) and r.fit_eligible_weeks==int(g.fit_eligible.sum()) and r.validation_accepted_weeks==int(g.validation_accepted.sum()) and r.changed_probability_weeks==int(g.changed_probability.sum()) and r.regressions==int(g['case'].eq('regression').sum()) and r.recoveries==int(g['case'].eq('recovery').sum())
    for r in csv('month_outcomes').itertuples():
        g=weekly[weekly.cutoff.eq(r.cutoff)&weekly.history.eq(r.history)&weekly.method.eq(r.method)];g=g if r.component=='all' else g[g.state.eq(r.component)];assert r.n==len(g);eq(float(((g.probability-g.actual_up)**2-(g.annual_probability-g.actual_up)**2).mean()) if len(g) else None,r.brier_difference);assert r.regressions==int(g['case'].eq('regression').sum()) and r.recoveries==int(g['case'].eq('recovery').sum())
    return dict(maximum_prediction_gap=maxgap,exact_fallback_seed_rows=fallback,independent_metric_cells=count)
def check_inference():
    e=csv('ensemble_predictions');pairs=read(OUT/'primary_comparisons.json');ps=[]
    for w in cfg()['windows']:
        n=w['n'];starts=np.random.default_rng(20260910).integers(n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);np.testing.assert_array_equal(ids,statistics.bootstrap_indices(n));p=e[e.date.between(w['start'],w['end'])]
        for r in [r for r in pairs if r['window']==w['name']]:
            a=p[p.history.eq(r['history'])&p.method.eq(r['candidate'])].sort_values('date');b=p[p.history.eq(r['reference_history'])&p.method.eq(r['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==n;v=(a.direction_up.to_numpy()!=a.actual_up.to_numpy()).astype(float)-(b.direction_up.to_numpy()!=b.actual_up.to_numpy()).astype(float) if r['metric']=='direction_error' else (a.probability.to_numpy()-a.actual_up.to_numpy())**2-(b.probability.to_numpy()-b.actual_up.to_numpy())**2;mean=float(v.mean());boot=v[ids].mean(1);center=(v-mean)[ids].mean(1);pval=(1+int((abs(center)>=abs(mean)).sum()))/10001;lo,hi=np.quantile(boot,[.025,.975]);eq(mean,r['difference']);eq(lo,r['ci95_low']);eq(hi,r['ci95_high']);assert pval==r['p'];ps.append(pval)
    adjusted=[0.]*len(ps);last=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):last=max(last,min(1.,ps[i]*(len(ps)-rank)));adjusted[i]=last
    np.testing.assert_array_equal(adjusted,[r['holm_adjusted_p'] for r in pairs]);assert len(pairs)==60
def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();start=time.time();last=read(OUT/'contract_verification.json')['completed_utc']
    for phase in ['fitting','validation','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    synthetic();ready=check_splits();sol,gates,fit=check_fits_and_gates();print('Independent purged splits, solvers, validation gates and information isolation PASS.',flush=True);pred=check_predictions();print('Independent forecast routing, fallback, metrics and quarter diagnostics PASS.',flush=True);check_inference();from audit_cadence40 import audit_cadence
    cadence=audit_cadence();assert old_evidence()==prep['old_evidence'];assert fit['independent_scalar_solutions']==read(OUT/'fitting_manifest.json')['scalar_fits'];files=[]
    for name,g in [('independent_solutions',sol),('independent_gate_decisions',gates)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],ready_months=ready,old_files_preserved=5554,independent_comparisons=60,**fit,**pred,**cadence,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('R40 independent verification PASS.',flush=True)
if __name__=='__main__':main()
