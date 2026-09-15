from common38 import *
from contract38 import synthetic,memberships
from verify37 import independent_metric
from scipy.special import expit
from collections import Counter
import math,itertools
def equal(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<tol,(a,b)
def overlap_checks():
    members=csv('membership');groups={k:g for k,g in members.groupby(['cutoff','view'])};count=0
    for row in csv('target_overlap').to_dict('records'):
        g=groups.get((row['cutoff'],row['view']),members.iloc[:0]);g=g if row['state']=='all' else g[g.state.eq(row['state'])];intervals=list(zip(g.entry.astype(int),g.exit.astype(int)));edges=Counter(i for a,b in intervals for i in range(a,b));n=len(g)
        pair=sum(max(a,c)<min(b,d) for (a,b),(c,d) in itertools.combinations(intervals,2));ordered=sorted(intervals,key=lambda x:(x[1],x[0]));dp=[0]
        for i,(a,b) in enumerate(ordered):
            previous=max([j+1 for j,(c,d) in enumerate(ordered[:i]) if d<=a]+[0]);dp.append(max(dp[-1],1+dp[previous]))
        expected=dict(n=n,unique_intervals=len(set(intervals)),total_edge_uses=sum(edges.values()),union_edges=len(edges),mean_edge_multiplicity=sum(edges.values())/len(edges) if edges else None,maximum_edge_multiplicity=max(edges.values(),default=0),overlapping_pairs=pair,maximum_disjoint_intervals=dp[-1])
        for k,v in expected.items():equal(v,row[k])
        count+=1
    return count
def observation_checks():
    obs,_,targets=data();o=csv('diagnostic_observations');member=csv('membership');mi=member.set_index(['cutoff','view','row_index']);models=csv('model_predictions');ensemble=csv('ensemble_predictions');models=models[models.history.eq(ANNUAL)&models.method.isin(LEARNED)].set_index(['row_index','method','seed']);ensemble=ensemble[ensemble.history.eq(ANNUAL)&ensemble.method.isin(LEARNED)].set_index(['row_index','method']);cache={}
    for t in read(OUT/'training_inputs.json'):
        d=arrays(t)
        for j,m in enumerate(LEARNED):
            for i,q in zip(d['row_index'],d['annual_logits'][:,j]):cache[(t['cutoff'],int(i),m,t['seed'])]=float(expit(q))
    gap=0.
    for r in o.itertuples():
        info=mi.loc[(r.cutoff,r.view,r.row_index)]
        for k in ['encoder_cutoff','year','date','weekday','state','entry','exit','joint_completed']:assert getattr(r,k)==info[k]
        if r.view=='future':p=float(ensemble.loc[(r.row_index,r.method),'probability']) if r.seed==-1 else float(models.loc[(r.row_index,r.method,r.seed),'probability'])
        else:p=math.fsum(cache[(r.cutoff,r.row_index,r.method,s)] for s in cfg()['seeds'])/3 if r.seed==-1 else cache[(r.cutoff,r.row_index,r.method,r.seed)]
        gap=max(gap,abs(p-r.probability));assert abs(p-r.probability)<1e-14;assert r.actual_up==int(targets['returns'][r.row_index]>0);equal(r.residual,r.actual_up-r.probability);assert r.score==r.probability and r.direction_up==int(r.probability>.5)
    expected=len(member)*16;assert len(o)==expected and not o.duplicated(['cutoff','view','row_index','method','seed']).any()
    return gap
def metric_checks():
    o=csv('diagnostic_observations');groups={k:g for k,g in o.groupby(['cutoff','view','method','seed'])};table=csv('cohort_metrics');count=0
    for row in table.to_dict('records'):
        g=groups.get(tuple(row[k] for k in ['cutoff','view','method','seed']),o.iloc[:0]);g=g if row['state']=='all' else g[g.state.eq(row['state'])];expected=independent_metric(g)
        for k,v in expected.items():equal(v,row[k])
        residual=math.fsum(int(y)-float(p) for y,p in zip(g.actual_up,g.probability));equal(residual,row['residual_sum']);equal(residual/len(g) if len(g) else None,row['residual_mean']);assert int(g.actual_up.sum())==row['up_n'] and len(g)-row['up_n']==row['down_n'] and row['opposite_class_pairs']==row['up_n']*row['down_n'];count+=1
    return count
def independent_agreement(a,b):
    if pd.isna(a) or pd.isna(b):return 'undefined'
    if abs(a)<=1e-12 or abs(b)<=1e-12:return 'near_zero'
    return 'same' if (a>0)==(b>0) else 'opposite'
def transfer_checks():
    metrics=csv('cohort_metrics').set_index(['cutoff','view','method','seed','state']);t=csv('residual_transfer');decomp=csv('residual_decomposition')
    for r in t.itertuples():
        for v in VIEWS:
            m=metrics.loc[(r.cutoff,v,r.method,r.seed,r.state)];assert getattr(r,v+'_n')==m.n;equal(getattr(r,v+'_residual'),m.residual_mean);equal(getattr(r,v+'_auroc'),m.auroc)
        assert r.r37_eligible==(r.daily_n>=20) and r.weekly_count_flag==(r.friday_n>=10 and r.future_n>=10)
        for a,b in [('daily','friday'),('daily','future'),('friday','future')]:assert getattr(r,a+'_'+b)==independent_agreement(getattr(r,a+'_residual'),getattr(r,b+'_residual'))
    for r in decomp.itertuples():
        a=[metrics.loc[(r.cutoff,r.source,r.method,r.seed,s)] for s in STATES];b=[metrics.loc[(r.cutoff,r.target,r.method,r.seed,s)] for s in STATES];sn=sum(v.n for v in a);tn=sum(v.n for v in b);assert sn==r.source_n and tn==r.target_n
        if not sn or not tn:
            assert r.status==('empty_source' if not sn else 'empty_target');assert pd.isna(r.overall_delta) and pd.isna(r.composition) and pd.isna(r.within_state);continue
        source_mean=math.fsum(v.residual_sum for v in a)/sn;target_mean=math.fsum(v.residual_sum for v in b)/tn;equal(target_mean-source_mean,r.overall_delta)
        if any(x.n==0 and y.n>0 for x,y in zip(a,b)):assert r.status=='target_state_absent_in_source' and pd.isna(r.composition) and pd.isna(r.within_state);continue
        standardized=math.fsum(y.n/tn*x.residual_mean for x,y in zip(a,b) if y.n);assert r.status=='defined';equal(standardized-source_mean,r.composition);equal(target_mean-standardized,r.within_state)
    for r in csv('transfer_summary').itertuples():
        p=next(p for p in periods() if p['name']==r.period);g=t[t.year.isin(period_years(p))&t.method.eq(r.method)&t.seed.eq(r.seed)];ids=[]
        for row in g.itertuples():
            if getattr(row,r.view+'_n')<=0 or row.future_n<=0:continue
            if r.cohort!='available' and row.daily_n<20:continue
            if r.cohort=='weekly_count10' and (row.friday_n<10 or row.future_n<10):continue
            ids.append(row)
        assert r.cells==len(ids) and r.future_weeks==sum(x.future_n for x in ids)
        for label in ['same','opposite','near_zero','undefined']:
            selected=[x for x in ids if independent_agreement(getattr(x,r.view+'_residual'),x.future_residual)==label];assert getattr(r,label+'_cells')==len(selected) and getattr(r,label+'_weeks')==sum(x.future_n for x in selected)
    return dict(transfer_cells=len(t),decomposition_cells=len(decomp),summary_cells=len(csv('transfer_summary')))
def accounting_checks():
    w=csv('weekly_brier_accounting');models=csv('model_predictions');e=csv('ensemble_predictions');e['seed']=-1;a=pd.concat([models,e],ignore_index=True);a=a[a.method.isin(LEARNED)&a.history.isin([ANNUAL]+list(POLICIES.values()))].set_index(['history','method','seed','row_index']);ctx=csv('weekly_context').set_index('row_index');maxgap=0.
    for r in w.itertuples():
        x=a.loc[(r.history,r.method,r.seed,r.row_index)];b=a.loc[(ANNUAL,r.method,r.seed,r.row_index)];c=ctx.loc[r.row_index];assert r.probability==x.probability and r.annual_probability==b.probability and r.actual_up==x.actual_up and r.eligible==(c.added_state_n>=20) and r.state==c.state and r.head_cutoff==c.head_cutoff
        delta=float(x.probability-b.probability);error=float(x.actual_up-b.probability);terms=dict(delta=delta,annual_residual=error,squared_shift=delta*delta,alignment_term=-2*delta*error,brier_change=(x.probability-x.actual_up)**2-(b.probability-x.actual_up)**2)
        for k,v in terms.items():equal(v,getattr(r,k))
        maxgap=max(maxgap,abs(r.squared_shift+r.alignment_term-r.brier_change));assert r.annual_correct==(b.direction_up==b.actual_up) and r.correct==(x.direction_up==x.actual_up)
        if not r.eligible:assert r.delta==0 and r.brier_change==0
    for r in csv('brier_summary').itertuples():
        p=next(p for p in periods() if p['name']==r.period);g=w[w.history.eq(r.history)&w.method.eq(r.method)&w.seed.eq(r.seed)&w.date.between(p['start'],p['end'])];g=g if r.scope=='all' else g[g.eligible];assert r.n==len(g)
        for key in ['squared_shift','alignment_term','brier_change']:equal(math.fsum(g[key])/len(g) if len(g) else None,getattr(r,key))
        assert r.regressions==int((g.annual_correct&~g.correct).sum()) and r.recoveries==int((~g.annual_correct&g.correct).sum())
    groups={k:g for k,g in w.groupby(['head_cutoff','history','method','seed','state'])};rank=csv('within_cell_ranking')
    for r in rank.itertuples():
        g=groups.get((r.cutoff,r.history,r.method,r.seed,r.state),w.iloc[:0]);pairs=list(itertools.combinations(range(len(g)),2));p=g.annual_probability.to_numpy();q=g.probability.to_numpy();inversions=ties=0
        for i,j in pairs:
            aa=int(p[i]>p[j])-int(p[i]<p[j]);bb=int(q[i]>q[j])-int(q[i]<q[j]);inversions+=aa*bb<0;ties+=(aa==0)!=(bb==0)
        assert r.n==len(g) and r.all_pairs==len(pairs) and r.inversions==inversions and r.tie_changes==ties
        baseline=g.copy();baseline['probability']=p;baseline['score']=p;baseline['direction_up']=(p>.5).astype(int);equal(independent_metric(baseline)['auroc'],r.annual_auroc);equal(independent_metric(g)['auroc'],r.corrected_auroc)
        if r.seed!=-1:assert r.inversions==0 and r.tie_changes==0;equal(r.annual_auroc,r.corrected_auroc)
    # Aggregate new accounting must reproduce the previous round's complete-period Brier changes.
    old=csv('ensemble_metrics').set_index(['period','history','method']);bs=csv('brier_summary')
    for r in bs[bs.seed.eq(-1)&bs.scope.eq('all')].itertuples():equal(old.loc[(r.period,r.history,r.method),'brier']-old.loc[(r.period,ANNUAL,r.method),'brier'],r.brier_change)
    return dict(weekly_accounting_rows=len(w),ranking_cells=len(rank),maximum_brier_identity_gap=maxgap,per_seed_rank_inversions=int(rank[rank.seed.ne(-1)].inversions.sum()),ensemble_rank_inversions=int(rank[rank.seed.eq(-1)].inversions.sum()))
def main():
    prep=check_frozen();phase=check_phase('diagnosis');assert not (OUT/'verification.json').exists();started=time.time();assert read(OUT/'contract_verification.json')['completed_utc']<=phase['started_utc']
    for k in ['protocol_sha256','source_sha256','input_sha256']:assert phase[k]==prep[k]
    synthetic();a,b=memberships();pd.testing.assert_frame_equal(a,csv('membership'),check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(b,csv('membership_checks'),check_dtype=False,check_exact=True);oc=overlap_checks();gap=observation_checks();print('Independent memberships, overlap intervals and probability sources PASS.',flush=True);mc=metric_checks();t=transfer_checks();print('Independent residual metrics, decompositions and sign tallies PASS.',flush=True);ac=accounting_checks();assert old_evidence()==prep['old_evidence']
    # Preserve complete archive copies byte-for-byte, including all previous accuracy results.
    for name in ['model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','primary_comparisons.json']:assert sha(OUT/name)==sha(V37/'results'/name)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5428,overlap_cells=oc,maximum_source_probability_gap=gap,metric_cells=mc,**t,**ac));print('R38 independent verification PASS.',flush=True)
if __name__=='__main__':main()
