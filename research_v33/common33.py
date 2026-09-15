"""Read-only accounting of changes in two already-frozen head pipelines."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V32=PROJECT/'research_v32'
sys.path.insert(0,str(V32));import common32 as previous_round
prior=previous_round.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;arrays=prior.arrays;data=prior.data;market=prior.market;gates=prior.gates;apply_pipeline=prior.apply_pipeline;training_rows=previous_round.training_rows
from scipy.special import expit
GROUPS=['intercept','decoded','market','vol','order'];METHODS=prior.LEARNED;PRIMARY=prior.PRIMARY;CASES=['regression','recovery','stable_correct','stable_wrong']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V32/'protocol.json']+[V32/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','heads.json','feature_banks.json','source_heads.json','routing.csv','membership.csv','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','yearly_metrics.csv','primary_comparisons.json']];r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V32/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V32/'results/delivery_manifest.json')['files'].items():r[str((V32/n).relative_to(PROJECT))]=d
    p=V32/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4895
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def group_indices(method):
    return [None,np.arange(25),np.arange(25,29),np.array([],int) if method=='learned_market' else np.array([29]),np.array([30]) if method in ['learned_order_extension','learned_order_offset'] else np.array([],int)]
def product_components(xa,ta,xn,tn,method):
    n=len(xa);old=np.zeros((n,5));new=old.copy();coef=old.copy();transform=old.copy();old[:,0]=ta[-1];new[:,0]=tn[-1];coef[:,0]=tn[-1]-ta[-1]
    for j,ids in enumerate(group_indices(method)[1:],1):
        if len(ids):
            old[:,j]=xa[:,ids]@ta[ids];new[:,j]=xn[:,ids]@tn[ids];coef[:,j]=((xn[:,ids]+xa[:,ids])*.5)@(tn[ids]-ta[ids]);transform[:,j]=(xn[:,ids]-xa[:,ids])@((tn[ids]+ta[ids])*.5)
    np.testing.assert_allclose(coef+transform,new-old,rtol=0,atol=1e-12);return old,new,coef,transform
def secant(za,zn):
    delta=zn-za;p=expit(za);w=p*(1-p);small=(abs(delta)<=1e-4)&(delta!=0);em=np.expm1(delta[small]);w[small]=p[small]*(1-p[small])*(em/delta[small])/(1+p[small]*em);large=abs(delta)>1e-4;w[large]=(expit(zn[large])-p[large])/delta[large];return w
def jobs():
    route=csv('routing');annual=read(OUT/'annual_heads.json');quarter=read(OUT/'quarter_heads.json');banks=read(OUT/'feature_banks.json');obs,price,_=data()
    for headcutoff,g in route.groupby('head_cutoff',sort=True):
        encoder=g.encoder_cutoff.iloc[0];rows=g.row_index.to_numpy(int);mf=market(price,obs,rows);gate=gates(price,obs,rows)
        for seed in cfg()['seeds']:
            bank=arrays(next(b for b in banks if b['cutoff']==encoder and b['seed']==seed));features=previous_round.bank_slice(bank,rows);old=[next(h for h in annual if h['cutoff']==encoder and h['seed']==seed and h['method']==m) for m in METHODS];new=old if headcutoff==encoder else [next(h for h in quarter if h['cutoff']==headcutoff and h['seed']==seed and h['method']==m) for m in METHODS];da=arrays(old[0]);dn=arrays(new[0]);xa=apply_pipeline(features,mf,gate,da);xn=apply_pipeline(features,mf,gate,dn)
            yield dict(head_cutoff=headcutoff,encoder_cutoff=encoder,seed=seed,rows=rows,features=features,market=mf,gate=gate,old_heads=old,new_heads=new,old_data=da,new_data=dn,old_x=xa,new_x=xn)
def parameter_changes(job):
    coef=[];trans=[]
    for old,new in zip(job['old_heads'],job['new_heads']):
        ta=np.asarray(old['coefficients']);tn=np.asarray(new['coefficients'])
        for group,ids in zip(GROUPS,group_indices(old['method'])):
            a=np.array([ta[-1]]) if ids is None else ta[ids];b=np.array([tn[-1]]) if ids is None else tn[ids];coef.append(dict(head_cutoff=job['head_cutoff'],encoder_cutoff=job['encoder_cutoff'],seed=job['seed'],method=old['method'],group=group,dimensions=len(a),old_norm=float(np.linalg.norm(a)),new_norm=float(np.linalg.norm(b)),delta_norm=float(np.linalg.norm(b-a)),old_intercept=float(ta[-1]) if group=='intercept' else np.nan,new_intercept=float(tn[-1]) if group=='intercept' else np.nan))
    da,dn=job['old_data'],job['new_data']
    for group,keys in [('rep',['lower','upper','mean','sd']),('market',['lower','upper','mean','sd']),('vol',['ray','projection','residual_mean','residual_sd']),('order',['ray','projection','residual_mean','residual_sd'])]:
        for key in keys:
            a=np.atleast_1d(da[group+'__'+key]);b=np.atleast_1d(dn[group+'__'+key]);trans.append(dict(head_cutoff=job['head_cutoff'],encoder_cutoff=job['encoder_cutoff'],seed=job['seed'],group=group,parameter=key,dimensions=a.size,old_norm=float(np.linalg.norm(a)),new_norm=float(np.linalg.norm(b)),delta_norm=float(np.linalg.norm(b-a)),maximum_absolute_change=float(np.max(abs(b-a))),maximum_abs_log_ratio=float(np.max(abs(np.log(b/a)))) if key in ['sd','residual_sd'] else np.nan))
    return coef,trans
def window_changes():
    obs,_,targets=data();summary=[];members=[]
    for cutoff in cfg()['decision_dates']:
        encoder=previous_round.annual_for(cutoff);a=set(training_rows(obs,encoder).tolist());b=set(training_rows(obs,cutoff).tolist());subsets={'retained':a&b,'added':b-a,'removed':a-b};row=dict(head_cutoff=cutoff,encoder_cutoff=encoder,annual_train_n=len(a),quarter_train_n=len(b),annual_up_fraction=float((targets['returns'][sorted(a)]>0).mean()),quarter_up_fraction=float((targets['returns'][sorted(b)]>0).mean()))
        for name,ids in subsets.items():
            ids=sorted(ids);row[name+'_n']=len(ids);row[name+'_up_fraction']=float((targets['returns'][ids]>0).mean()) if ids else np.nan;row[name+'_mean_return']=float(targets['returns'][ids].mean()) if ids else np.nan
            for i in ids:members.append(dict(head_cutoff=cutoff,encoder_cutoff=encoder,role=name,row_index=i,date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],actual_return=float(targets['returns'][i]),actual_up=int(targets['returns'][i]>0)))
        row['up_fraction_change']=row['quarter_up_fraction']-row['annual_up_fraction'];summary.append(row)
    return pd.DataFrame(summary),pd.DataFrame(members)
def weekly_from(seed):
    # Per-seed components are unlabelled until they are checked against the archive.
    archive=csv('ensemble_predictions');lookup={(r.history,r.method,r.date):r for r in archive.itertuples()};records=[]
    for (method,date),g in seed.groupby(['method','date'],sort=True):
        assert len(g)==15;ref=lookup[(cfg()['reference'],method,date)];cand=lookup[(cfg()['candidate'],method,date)];old_correct=ref.direction_up==ref.actual_up;new_correct=cand.direction_up==cand.actual_up;case='regression' if old_correct and not new_correct else 'recovery' if new_correct and not old_correct else 'stable_correct' if old_correct else 'stable_wrong';direction=2*ref.actual_up-1
        row=dict(method=method,date=date,year=int(date[:4]),row_index=int(ref.row_index),joint_completed=ref.joint_completed,head_cutoff=g.head_cutoff.iloc[0],encoder_cutoff=g.encoder_cutoff.iloc[0],actual_up=int(ref.actual_up),actual=float(ref.actual),old_probability=ref.probability,new_probability=cand.probability,probability_change=cand.probability-ref.probability,old_up=int(ref.direction_up),new_up=int(cand.direction_up),old_correct=old_correct,new_correct=new_correct,case=case,old_margin=ref.probability-.5,new_margin=cand.probability-.5,brier_change=(cand.probability-ref.actual_up)**2-(ref.probability-ref.actual_up)**2)
        values=[]
        for group in GROUPS:
            h=g[g.group.eq(group)];assert sorted(h.seed)==cfg()['seeds'];p=float(h.probability_contribution.mean());c=float(h.coefficient_probability_contribution.mean());t=float(h.transform_probability_contribution.mean());row['p_'+group]=p;row['coef_'+group]=c;row['transform_'+group]=t;row['signed_'+group]=direction*p;values.append(direction*p)
        row['signed_probability_change']=direction*row['probability_change'];index=int(np.argmin(values));row['dominant_adverse_group']=GROUPS[index] if values[index]<-1e-12 else 'none';row['dominant_ties']=int(np.sum(np.abs(np.asarray(values)-values[index])<=1e-12)) if values[index]<-1e-12 else 0
        assert abs(sum(row['p_'+k] for k in GROUPS)-row['probability_change'])<2e-13;records.append(row)
    return pd.DataFrame(records)
def summaries(weekly):
    selections=[(w['name'],weekly[weekly.date.between(w['start'],w['end'])]) for w in cfg()['windows']]+[('pooled_2021_2026',weekly)]+[(f'year_{year}',g) for year,g in weekly.groupby('year')];performance=[];contributions=[];dominant=[]
    for period,frame in selections:
        for method,g in frame.groupby('method',sort=True):
            row=dict(period=period,method=method,n=len(g),annual_correct=int(g.old_correct.sum()),quarter_head_correct=int(g.new_correct.sum()),annual_accuracy=float(g.old_correct.mean()),quarter_head_accuracy=float(g.new_correct.mean()),annual_brier=float(((g.old_probability-g.actual_up)**2).mean()),quarter_head_brier=float(((g.new_probability-g.actual_up)**2).mean()))
            for case in CASES:row[case]=int(g.case.eq(case).sum())
            performance.append(row)
            for case in ['all']+CASES:
                s=g if case=='all' else g[g.case.eq(case)];sign=2*s.actual_up-1
                for group in GROUPS:
                    contributions.append(dict(period=period,method=method,case=case,group=group,n=len(s),mean_probability_contribution=float(s['p_'+group].mean()) if len(s) else np.nan,mean_signed_contribution=float(s['signed_'+group].mean()) if len(s) else np.nan,mean_signed_coefficient=float((sign*s['coef_'+group]).mean()) if len(s) else np.nan,mean_signed_transform=float((sign*s['transform_'+group]).mean()) if len(s) else np.nan))
                    dominant.append(dict(period=period,method=method,case=case,group=group,n=len(s),dominant_adverse_count=int(s.dominant_adverse_group.eq(group).sum())))
    return dict(performance_summary=pd.DataFrame(performance),component_summary=pd.DataFrame(contributions),dominant_summary=pd.DataFrame(dominant))
