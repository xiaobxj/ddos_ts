"""Full H5 quarterly policy and independent causal calibration banks."""
from experiment3 import *
from scipy.special import expit,logit
v2cal=load_local('v2_calibration_for_v3',V2/'calibration.py')
v2score=load_local('v2_score_for_v3',V2/'score.py')
metric=v2score.metric

def all_cutoffs(cadence):
    if cadence=='annual':return [f'{y}-12-31' for y in range(2022,2026)]
    return pd.date_range('2022-12-31','2026-06-30',freq='QE').strftime('%Y-%m-%d').tolist()

def raw_banks():
    f=data();obs,_=observations(f,5);base=load_csv(V2/'results/context_h5.csv').set_index('date')
    banks=[];seeds=[]
    for cadence in ['annual','quarterly']:
        for c in all_cutoffs(cadence):
            folder=fit_folder(c);done=read(folder/'completed.json')
            for p,s in done['files'].items():assert sha(folder/p)==s,(folder,p)
            end=(pd.Timestamp(c)+pd.offsets.QuarterEnd()).strftime('%Y-%m-%d') if cadence=='quarterly' else f'{int(c[:4])+1}-12-31'
            ids=np.flatnonzero(obs.date.gt(c)&obs.date.le(end));context=obs.iloc[ids].copy();context['h']=5
            context['state']=context.date.map(base.state).to_numpy(int);context['cutoff']=c;context['cadence']=cadence
            with np.load(folder/'controls.npz') as z:market=z['market_probability'][ids]
            for method in METHODS:
                probs=[]
                for seed in SEEDS:
                    with np.load(folder/f'cache_{seed}_e20.npz') as z:p=z['p__'+method+'.U'][ids]
                    s=context.copy();s['method']=method;s['seed']=seed;s['probability']=p;seeds.append(s);probs.append(p)
                b=context.copy();b['method']=method;b['probability']=np.mean(probs,axis=0);banks.append(b)
            for name,p in [('frequency',np.full(len(ids),done['training_frequency'])),('market4',market)]:
                b=context.copy();b['method']=name;b['probability']=p;banks.append(b)
    return pd.concat(banks,ignore_index=True),pd.concat(seeds,ignore_index=True)

def shrink_decision(bank,cutoff):
    tr,va,ready=v2cal.split_bank(bank,cutoff);q=float(tr.actual_up.mean()) if len(tr) else .5
    alpha=1.;candidate_alpha=1.;keep=False;delta=None;wins=None;reason='history_insufficient'
    nt=maximal_nonoverlap(tr);nv=maximal_nonoverlap(va)
    if cutoff[5:7]=='12':reason='annual_reset'
    elif ready and nt>=10 and nv>=5:
        p=tr.probability.to_numpy(float);y=tr.actual_up.to_numpy(float);d=p-q
        denom=float(d@d);candidate_alpha=float(np.clip(d@(y-q)/denom,0,1)) if denom>1e-15 else 0.
        v=q+candidate_alpha*(va.probability.to_numpy()-q)
        keep,delta,wins=v2cal.accepted(va.actual_up.to_numpy(),v,va.probability.to_numpy())
        reason='accepted' if keep else 'validation_quality_rejected'
        if keep:alpha=candidate_alpha
    return dict(cutoff=cutoff,scheme='shrink',accepted=bool(keep),reason=reason,q=q,alpha=alpha,candidate_alpha=candidate_alpha,
        train_n=len(tr),validation_n=len(va),train_disjoint=nt,validation_disjoint=nv,
        validation_brier_delta=delta,validation_correct_delta=wins,
        training_dates=tr.date.tolist(),validation_dates=va.date.tolist(),
        train_last_maturity=tr.joint_completed.max() if len(tr) else '',validation_first=va.date.min() if len(va) else '')

def cal_record(bank,cutoff,scheme):
    if scheme=='shrink':return shrink_decision(bank,cutoff)
    params,rows=v2cal.decision(bank,cutoff,'quarter_platt');row=rows[0]
    row=dict(row,scheme='platt');row['slope_applied'],row['intercept_applied']=params.get(-1,(1.,0.))
    return row

def apply_record(p,record):
    p=np.asarray(p,float)
    if not record['accepted']:return p.copy()
    if record['scheme']=='shrink':return record['q']+record['alpha']*(p-record['q'])
    return expit(record['slope_applied']*logit(np.clip(p,1e-12,1-1e-12))+record['intercept_applied'])

def calibrated_banks(raw):
    outputs=[];records=[];ends=all_cutoffs('quarterly')
    for (cadence,method),bank in raw.groupby(['cadence','method'],sort=False):
        bank=bank.sort_values('date').reset_index(drop=True);b=bank.copy();b['scheme']='raw';outputs.append(b)
        if method not in METHODS:continue
        for scheme in ['platt','shrink']:
            b=bank.copy();b['scheme']=scheme;b['adjusted']=False
            for j,c in enumerate(ends):
                end=ends[j+1] if j+1<len(ends) else cfg()['data_end'];row=cal_record(bank,c,scheme)
                use=b.date.gt(c)&b.date.le(end);b.loc[use,'probability']=apply_record(bank.loc[use,'probability'],row)
                b.loc[use,'adjusted']=row['accepted'];row.update(cadence=cadence,method=method)
                row['actual_used_dates']=b.loc[use,'date'].tolist() if row['accepted'] else []
                records.append(row)
            outputs.append(b)
        # Training-bank frequency as a low-complexity calibration comparator.
        if method=='vol':
            b=bank.copy();b['method']='cal_frequency';b['scheme']='raw'
            for j,c in enumerate(ends):
                tr,va,ready=v2cal.split_bank(bank,c);end=ends[j+1] if j+1<len(ends) else cfg()['data_end']
                if ready and c[5:7]!='12':b.loc[b.date.gt(c)&b.date.le(end),'probability']=tr.actual_up.mean()
            outputs.append(b)
    result=pd.concat(outputs,ignore_index=True)
    result['name']=result.cadence+'.'+result.method+'.'+result.scheme
    return result,records

def evaluate(p):
    rows=[];years=[];states=[];bins=[]
    for name,g in p[p.matured].groupby('name',sort=False):
        rows.append(dict(name=name,**metric(g.actual_up,g.probability)))
        for year,a in g.groupby('year'):years.append(dict(name=name,year=int(year),**metric(a.actual_up,a.probability)))
        for state,a in g.groupby('state'):states.append(dict(name=name,state=int(state),**metric(a.actual_up,a.probability)))
        b=np.minimum((g.probability.to_numpy()*10).astype(int),9)
        for k in range(10):
            a=g[b==k];bins.append(dict(name=name,lower=k/10,upper=(k+1)/10,n=len(a),
                p_mean=float(a.probability.mean()) if len(a) else None,up_rate=float(a.actual_up.mean()) if len(a) else None))
    return rows,years,states,bins

def bootstrap(diff,block):
    rng=np.random.default_rng(603987+block);n=len(diff);vals=[]
    for _ in range(20):
        starts=rng.integers(0,n,(500,int(np.ceil(n/block))));ids=(starts[:,:,None]+np.arange(block))%n;ids=ids.reshape(500,-1)[:,:n]
        vals.extend(np.asarray(diff)[ids].mean(1))
    vals=np.asarray(vals);mean=float(np.mean(diff))
    return dict(n=n,block=block,delta_brier=mean,ci_low=float(np.quantile(vals,.025)),ci_high=float(np.quantile(vals,.975)),
        p_two_sided=float((1+(abs(vals-mean)>=abs(mean)).sum())/(len(vals)+1)))

def main():
    check_freeze()
    for lane in [0,1]:assert read(OUT/f'training_lane{lane}_completed.json')['status']=='PASS'
    for p,s in read(OUT/'evaluation_freeze.json')['files'].items():assert sha(ROOT/p)==s,p
    raw,seeds=raw_banks();p,records=calibrated_banks(raw)
    obs,_=observations(data(),5);dates=obs[obs.matured&obs.date.ge('2023-01-01')].date.tolist()
    for name,g in p[p.matured].groupby('name'):assert sorted(g.date)==dates and g.date.is_unique,name
    rows,years,states,bins=evaluate(p)
    stats=[]
    for a,b in cfg()['slow']['primary_comparisons']:
        x=p[p.name.eq(a)&p.matured].sort_values('date');y=p[p.name.eq(b)&p.matured].sort_values('date')
        diff=(x.probability.to_numpy()-x.actual_up.to_numpy())**2-(y.probability.to_numpy()-y.actual_up.to_numpy())**2
        for block in [20,40]:stats.append(dict(candidate=a,control=b,**bootstrap(diff,block)))
    for block in [20,40]:
        family=sorted([s for s in stats if s['block']==block],key=lambda x:x['p_two_sided']);running=0.
        for i,s in enumerate(family):running=max(running,min(1.,(4-i)*s['p_two_sided']));s['p_holm']=running
    csv(OUT/'slow_raw_banks.csv',raw);csv(OUT/'slow_seed_predictions.csv',seeds);csv(OUT/'slow_predictions.csv',p)
    for name,values in [('slow_metrics',rows),('slow_years',years),('slow_states',states),('slow_reliability',bins),('slow_bootstrap',stats)]:csv(OUT/(name+'.csv'),pd.DataFrame(values))
    save(OUT/'calibration_records.json',records)
    latest=p[p.date.eq(cfg()['data_end'])][['name','probability','cutoff']].to_dict('records');save(OUT/'slow_latest.json',dict(computed_utc=now(),rows=latest))
    save(OUT/'slow_scored.json',dict(status='PASS',completed_utc=now(),n=len(dates),candidates=len(rows),records=len(records)))

if __name__=='__main__':main()
