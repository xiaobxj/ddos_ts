from common36 import *
from contract36 import synthetic_cases,checks
import diagnose36 as diagnosis
import math

def quantile(values,q):
    x=sorted(float(v) for v in values);p=q*(len(x)-1);lo=math.floor(p);hi=math.ceil(p);return x[lo]+(p-lo)*(x[hi]-x[lo])
def close(a,b,tol=1e-12):
    if pd.isna(a) or pd.isna(b):assert pd.isna(a) and pd.isna(b),(a,b)
    else:assert abs(a-b)<=tol*max(1,abs(a),abs(b)),(a,b)
def independent_coverage(x,lo,hi,mean,sd):
    ans=[]
    for r in np.asarray(x,float):
        flags=[float(v)<float(l) or float(v)>float(h) for v,l,h in zip(r,lo,hi)];z=[(float(v)-float(m))/float(s) for v,m,s in zip(r,mean,sd)];ans.append([float(any(flags)),sum(flags)/len(flags),math.fsum(v*v for v in z)/len(z)])
    return np.asarray(ans)
def sparse_metrics(g,r):
    n=len(g);assert n==r.n and r.supported==(n>=cfg()['support']['weekly_cell_min'])
    if n==0:assert r.correct_directions==0 and pd.isna(r.accuracy) and pd.isna(r.brier);return
    y=g.actual_up.to_numpy(bool);up=g.probability.to_numpy()>.5;p=g.probability.to_numpy();count=int((y==up).sum());assert count==r.correct_directions;close(count/n,r.accuracy);close(((p-y)**2).mean(),r.brier);clip=np.clip(p,1e-12,1-1e-12);close(-np.where(y,np.log(clip),np.log1p(-clip)).mean(),r.log_loss);close(up.mean(),r.predicted_up_fraction);close(y.mean(),r.observed_up_fraction)
    if y.all() or not y.any():assert pd.isna(r.auroc) and pd.isna(r.balanced_accuracy)
    else:
        scores=p[y,None]-p[~y][None,:];close(((scores>0)+.5*(scores==0)).mean(),r.auroc);close((up[y].mean()+(~up[~y]).mean())/2,r.balanced_accuracy)

def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();started=time.time();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['calibration','diagnosis']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    assert len(synthetic_cases())==8;obs,price=raw_data();raw=descriptor_matrix(obs,price);states=csv('state_observations');dc=csv('decoded_coverage_by_seed');thresholds=read(OUT/'state_thresholds.json');heads=read(OUT/'annual_heads.json');banks=read(OUT/'feature_banks.json');maxcov=0.;temporal=[]
    for t in thresholds:
        annual=t['encoder_cutoff'];tr=training_rows(obs,annual);x=raw[tr];close(t['volatility_median'],quantile(x[:,1],.5));assert t['train_n']==len(tr) and t['maximum_maturity']==obs.joint_completed.iloc[tr].max()<=annual
        for j in range(5):
            mean=math.fsum(x[:,j])/len(x);sd=math.sqrt(math.fsum((float(v)-mean)**2 for v in x[:,j])/len(x));close(t['lower'][j],quantile(x[:,j],.01));close(t['upper'][j],quantile(x[:,j],.99));close(t['mean'][j],mean);close(t['sd'][j],max(sd,1e-6))
        poison=raw.copy();poison[~np.isin(np.arange(len(obs)),tr)]=999999.;assert calibrate(raw,tr)==calibrate(poison,tr)
        g=states[states.encoder_cutoff.eq(annual)].sort_values('row_index');rows=g.row_index.to_numpy(int);np.testing.assert_allclose(g[NAMES].to_numpy(),raw[rows],rtol=1e-13,atol=1e-13)
        labels=[('negative_' if r[0]<0 else 'nonnegative_')+('low' if r[1]<=t['volatility_median'] else 'high') for r in raw[rows]];assert g.state.tolist()==labels
        cv=independent_coverage(raw[rows],t['lower'],t['upper'],t['mean'],t['sd']);cols=['outside_any','outside_fraction','mean_squared_raw_z']
        for j,n in enumerate(cols):gap=float(abs(cv[:,j]-g['market5_'+n]).max());maxcov=max(maxcov,gap);np.testing.assert_allclose(cv[:,j],g['market5_'+n],rtol=1e-11,atol=1e-12)
        flags=(raw[rows]<np.array(t['lower']))|(raw[rows]>np.array(t['upper']))
        for j,n in enumerate(NAMES):np.testing.assert_array_equal(flags[:,j],g['outside_'+n])
        allcov=[]
        for seed in cfg()['seeds']:
            h=next(h for h in heads if h['cutoff']==annual and h['seed']==seed and h['method']==LEARNED[0]);d=arrays(h);bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));f=bank_slice(bank,rows);cv=independent_coverage(f,d['rep__lower'],d['rep__upper'],d['rep__mean'],d['rep__sd']);part=dc[dc.encoder_cutoff.eq(annual)&dc.seed.eq(seed)].sort_values('row_index');np.testing.assert_array_equal(part.row_index,rows)
            for j,n in enumerate(cols):gap=float(abs(cv[:,j]-part['decoded25_'+n]).max());maxcov=max(maxcov,gap);np.testing.assert_allclose(cv[:,j],part['decoded25_'+n],rtol=1e-11,atol=1e-12)
            allcov.append(cv)
        for j,n in enumerate(cols):np.testing.assert_allclose(np.mean(allcov,axis=0)[:,j],g['decoded25_'+n],rtol=1e-11,atol=1e-12)
        temporal.append(dict(encoder_cutoff=annual,annual_n=len(tr),excluded_descriptors_poisoned=int(len(obs)-len(tr)),thresholds_unchanged=True))
    # Check inherited descriptor functions against future-bar perturbations at every decision segment.
    for cutoff,g in csv('routing').groupby('head_cutoff'):
        row=int(g.row_index.max());anchor=int(obs.anchor.iloc[row]);poison=price.copy();cols=['open','high','low','close','volume'];poison.loc[poison.index>anchor,cols]=poison.loc[poison.index>anchor,cols]*1.73
        idx=np.array([row]);a=np.r_[prior.market(price,obs,idx)[0],prior.gates(price,obs,idx)[0]];b=np.r_[prior.market(poison,obs,idx)[0],prior.gates(poison,obs,idx)[0]];np.testing.assert_array_equal(a,b)
    _,_,targets=prior.data();y=(targets['returns']>0).astype(int);summary=csv('training_state_summary');comparisons=csv('state_comparisons');decomp=csv('label_mix_decomposition');sc=states.set_index(['encoder_cutoff','row_index'])
    for r in summary.itertuples():
        ids=roles(obs,r.cutoff)[r.role];g=sc.loc[[(r.encoder_cutoff,int(i)) for i in ids]] if len(ids) else states.iloc[:0];mask=g.state.eq(r.state).to_numpy();sel=ids[mask];assert r.role_n==len(ids) and r.n==len(sel) and r.up_n==int(y[sel].sum()) and r.supported==(len(sel)>=20)
        close(r.share,len(sel)/len(ids) if len(ids) else np.nan);close(r.up_rate,y[sel].mean() if len(sel) else np.nan)
        for name in diagnosis.COVERAGE+NAMES:close(getattr(r,'mean_'+name),math.fsum(g.loc[mask,name])/len(sel) if len(sel) else np.nan)
    lookup=summary.set_index(['cutoff','role','state'])
    for r in comparisons.itertuples():
        a=lookup.loc[(r.cutoff,'annual',r.state)];t=lookup.loc[(r.cutoff,r.target_role,r.state)];assert r.annual_n==a.n and r.target_n==t.n and r.supported==bool(a.supported and t.supported);close(r.share_delta,t.share-a.share);close(r.up_rate_delta,t.up_rate-a.up_rate)
        for n in diagnosis.COVERAGE:close(getattr(r,n+'_delta'),t['mean_'+n]-a['mean_'+n])
    for r in decomp.itertuples():
        a=summary[summary.cutoff.eq(r.cutoff)&summary.role.eq('annual')].set_index('state');t=summary[summary.cutoff.eq(r.cutoff)&summary.role.eq(r.target_role)].set_index('state');at=int(a.n.sum());nt=int(t.n.sum())
        assert r.annual_n==at and r.target_n==nt
        if nt==0:assert r.status=='empty_target' and pd.isna(r.overall_delta);continue
        delta=float(t.up_n.sum()/nt-a.up_n.sum()/at);close(delta,r.overall_delta)
        if any(a.loc[s,'n']==0 and t.loc[s,'n']>0 for s in STATES):assert r.status=='target_state_absent_in_annual';continue
        predicted_target_using_annual=math.fsum((t.loc[s,'n']/nt)*(a.loc[s,'up_n']/a.loc[s,'n'] if a.loc[s,'n'] else 0) for s in STATES)
        close(r.composition,predicted_target_using_annual-a.up_n.sum()/at);close(r.within_state,t.up_n.sum()/nt-predicted_target_using_annual);close(r.composition+r.within_state,delta)
    context=csv('weekly_context');assert len(context)==272
    for r in context.itertuples():
        assert r.head_cutoff<r.date and r.encoder_cutoff<=r.head_cutoff and r.state==sc.loc[(r.encoder_cutoff,r.row_index),'state'];a=lookup.loc[(r.head_cutoff,'annual',r.state)];n=lookup.loc[(r.head_cutoff,'added',r.state)];assert r.annual_state_n==a.n and r.added_state_n==n.n and r.training_support==bool(a.supported and n.supported);close(r.added_state_up_rate,n.up_rate);close(r.annual_state_up_rate,a.up_rate)
    e=csv('ensemble_predictions').merge(context[['row_index','state']],on='row_index',validate='many_to_one');m=csv('model_predictions').merge(context[['row_index','state']],on='row_index',validate='many_to_one')
    for name,source in [('state_prediction_metrics',e),('state_seed_metrics',m)]:
        for r in csv(name).itertuples():
            p=next(p for p in periods() if p['name']==r.period);g=source[source.date.between(p['start'],p['end'])&source.state.eq(r.state)&source.history.eq(r.history)&source.method.eq(r.method)]
            if name=='state_seed_metrics':g=g[g.seed.eq(r.seed)]
            sparse_metrics(g,r)
    w=csv('weekly_state_effects');old=csv('archived_weekly_effects');pd.testing.assert_frame_equal(w[old.columns],old,check_dtype=False,check_exact=True);assert len(w)==1088
    for r in csv('state_effect_summary').itertuples():
        p=next(p for p in periods() if p['name']==r.period);g=w[w.date.between(p['start'],p['end'])&w.state.eq(r.state)&w.method.eq(r.method)];g=g if r.case=='all' else g[g['case'].eq(r.case)];assert r.n==len(g) and r.supported==(len(g)>=10) and r.training_supported_weeks==int(g.training_support.sum())
        for n in EFFECTS:
            close(getattr(r,'mean_'+n),math.fsum(g[n])/len(g) if len(g) else np.nan);close(getattr(r,'mean_signed_'+n),math.fsum(g['signed_'+n])/len(g) if len(g) else np.nan)
    # Each state partition recovers the unchanged whole-period result without double counting.
    sm=csv('state_prediction_metrics')
    for (period,history,method),g in sm.groupby(['period','history','method']):
        p=next(p for p in periods() if p['name']==period);original=e[e.date.between(p['start'],p['end'])&e.history.eq(history)&e.method.eq(method)];assert g.n.sum()==len(original) and g.correct_directions.sum()==int(original.direction_up.eq(original.actual_up).sum());close((g.brier.fillna(0)*g.n).sum()/g.n.sum(),((original.probability-original.actual_up)**2).mean())
    # Independent arithmetic above precedes serialization/join replay.
    ts,comp,dec=diagnosis.training_summaries();ctx=diagnosis.contexts(ts);pm,ps=diagnosis.prediction_tables(ctx);ws,eff=diagnosis.effect_tables(ctx)
    latest=[ts[ts.encoder_cutoff.eq(a)].cutoff.max() for a in sorted(ts.encoder_cutoff.unique())]
    for name,g in [('training_state_summary',ts),('state_comparisons',comp),('label_mix_decomposition',dec),('latest_annual_snapshot',ts[ts.cutoff.isin(latest)]),('weekly_context',ctx),('state_prediction_metrics',pm),('state_seed_metrics',ps),('weekly_state_effects',ws),('state_effect_summary',eff),('all_2026_cases',ws[ws.year.eq(2026)]),('regressions_2026',ws[ws.year.eq(2026)&ws['case'].eq('regression')])]:pd.testing.assert_frame_equal(g.reset_index(drop=True),csv(name),check_dtype=False,atol=1e-12,rtol=1e-12)
    assert old_evidence()==prep['old_evidence'];check_frozen();p=OUT/'temporal_checks.csv';pd.DataFrame(temporal).to_csv(p,index=False)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5250,annual_calibrations=6,descriptor_future_bar_checks=23,training_state_cells=460,forecast_state_metric_cells=576,seed_state_metric_cells=1728,weekly_method_cases=1088,maximum_coverage_gap=maxcov,independent_quantiles=True,independent_coverage=True,independent_label_mix_accounting=True,sparse_metrics_verified=True,archived_predictions_unchanged=True,new_head_fits=0,new_neural_fits=0,new_predictions=0,new_inferential_tests=0,artifacts={str(p.relative_to(ROOT)):sha(p)}));print('Verification PASS: causal states/coverage, labels, sparse metrics and unchanged predictions.',flush=True)
if __name__=='__main__':main()
