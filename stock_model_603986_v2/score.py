"""Score all frozen candidates only after both horizon training manifests exist."""
from core import *
from calibration import calibrate
from scipy.special import expit
from scipy.stats import rankdata

def metric(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);up=p>.5
    assert len(y) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    n1=int(y.sum());n0=len(y)-n1
    auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
    ba=float(((up[y==1]).mean()+(~up[y==0]).mean())/2) if n1 and n0 else None
    clipped=np.clip(p,1e-12,1-1e-12)
    return dict(n=len(y),correct=int((up==y).sum()),accuracy=float((up==y).mean()),balanced_accuracy=ba,
        auc=auc,brier=float(np.mean((p-y)**2)),logloss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
        actual_up_frequency=float(y.mean()),predicted_up_fraction=float(up.mean()),probability_mean=float(p.mean()),
        probability_sd=float(p.std()),probability_min=float(p.min()),probability_max=float(p.max()))

def build(h):
    f=data();obs,_=observations(f,h);r,_,_=modules();rows=[];seedrows=[];contexts=[];native=[]
    for task in tasks(h):
        folder=OUT/'fits'/task_id(task);done=read(folder/'completed.json')
        for p,s in done['files'].items():assert sha(folder/p)==s,p
        if task['cadence']=='annual':end=str(int(task['cutoff'][:4])+1)+'-12-31'
        else:end=(pd.Timestamp(task['cutoff'])+pd.offsets.QuarterEnd()).strftime('%Y-%m-%d')
        mask=obs.date.gt(task['cutoff'])&obs.date.le(end);idx=np.flatnonzero(mask)
        with np.load(folder/'controls.npz') as z:controls={k:z[k] for k in z.files}
        context=obs.loc[mask].copy();context['h']=h;context['training_cutoff']=task['cutoff']
        context['state']=(controls['market'][idx,0]>=0).astype(int)*2+(controls['market'][idx,1]>done['volatility_median']).astype(int)
        ismain=task['objective']=='mse' and task['years']==5
        if ismain and task['cadence']=='annual':contexts.append(context)
        schedule='q2026' if task['cadence']!='annual' else f"{task['years']}y"
        for epoch in [10,20]:
            prob={};native_scores=[]
            for seed in SEEDS:
                with np.load(folder/f'cache_{seed}_e{epoch}.npz') as z:
                    for key in z.files:
                        if key.startswith('p__'):prob.setdefault(key[3:],[]).append(z[key][idx])
                    native_scores.append(z['all_native'][idx])
                prefix=f"{task['objective']}_{schedule}_e{epoch}"
                for key,ps in prob.items():
                    sr=context.copy();sr['method']=prefix+'.'+key;sr['seed']=seed;sr['probability']=ps[-1];seedrows.append(sr)
            for key,ps in prob.items():
                result=context.copy();result['method']=prefix+'.'+key;result['probability']=np.mean(ps,axis=0);rows.append(result)
            if task['objective']=='mse':
                scale=read(folder/'scales.json');nr=context.copy();nr['method']=prefix+'.native_return'
                nr['predicted_return']=np.mean(native_scores,axis=0)*scale['returns_sd']+scale['returns_mean'];native.append(nr)
        if task['objective']=='mse':
            for name,p in [('frequency',np.full(len(idx),done['training_frequency'])),('market4',controls['market_probability'][idx])]:
                result=context.copy();result['method']=name+'_'+schedule;result['probability']=p;rows.append(result)
    context=pd.concat(contexts).sort_values('date').reset_index(drop=True)
    allrows=pd.concat(rows,ignore_index=True)
    # Define a full-period quarterly policy: annual through Q1, most recent quarterly thereafter.
    annual=allrows[allrows.method.str.startswith('mse_5y_')|allrows.method.isin(['frequency_5y','market4_5y'])]
    for name,g in annual.groupby('method',sort=False):
        qname=name.replace('5y','q2026');copy=g[g.date.le('2026-03-31')].copy();copy['method']=qname;allrows=pd.concat([allrows,copy],ignore_index=True)
    for name,p in [('constant_0.5',.5),('always_up',1.),('always_down',0.)]:
        g=context.copy();g['method']=name;g['probability']=p;allrows=pd.concat([allrows,g],ignore_index=True)
    seeds=pd.concat(seedrows,ignore_index=True);native=pd.concat(native,ignore_index=True)
    for table in [seeds,native]:
        base=table[table.method.str.startswith('mse_5y_')&table.date.le('2026-03-31')].copy()
        base['method']=base.method.str.replace('5y','q2026',regex=False)
        if table is seeds:seeds=pd.concat([table,base],ignore_index=True)
        else:native=pd.concat([table,base],ignore_index=True)
    # State-stratified evaluation uses one common, causal annual reference for all arms.
    state_map=context.set_index('date').state
    for table in [allrows,seeds,native]:
        table['model_state']=table.state;table['state']=table.date.map(state_map).astype(int)
    bank=allrows[allrows.method.eq(cfg()['primary'])].sort_values('date').reset_index(drop=True)
    cal,decisions,reuse=calibrate(bank);allrows=pd.concat([allrows,cal],ignore_index=True)
    return allrows,seeds,native,context,decisions,reuse

def bootstrap_delta(delta,h,control,block):
    n=len(delta);rng=np.random.default_rng(cfg()['evaluation']['bootstrap_seed']+h+block)
    draws=cfg()['evaluation']['bootstrap_draws'];values=[]
    # Circular moving blocks. Same dates and targets paired within every resample.
    for lo in range(0,draws,500):
        start=rng.integers(0,n,size=(min(500,draws-lo),int(np.ceil(n/block))))
        indices=(start[:,:,None]+np.arange(block))%n;indices=indices.reshape(len(start),-1)[:,:n]
        values.extend(np.asarray(delta)[indices].mean(1).tolist())
    values=np.asarray(values);mean=float(np.mean(delta));centered=values-mean
    p=float((1+np.sum(abs(centered)>=abs(mean)))/(len(values)+1))
    return dict(h=h,control=control,block=block,n=n,delta_brier=mean,
        ci_low=float(np.quantile(values,.025)),ci_high=float(np.quantile(values,.975)),p_two_sided=p)

def main():
    check_freeze()
    for name,digest in read(OUT/'evaluation_implementation_freeze.json')['files'].items():assert sha(ROOT/name)==digest,name
    for h in [1,5]:assert read(OUT/f'training_h{h}_completed.json')['status']=='PASS'
    assert not (OUT/'scoring_started.json').exists()
    save(OUT/'scoring_started.json',dict(started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),
        evaluation_sources={p.name:sha(p) for p in [ROOT/'score.py',ROOT/'calibration.py']}))
    metrics=[];yearly=[];states=[];reliability=[];boots=[];counts=[];latest=[];native_metrics=[];seedmetrics=[];updates=[]
    for h in [1,5]:
        allrows,seeds,native,context,decisions,reuse=build(h)
        csv(OUT/f'predictions_h{h}.csv',allrows);csv(OUT/f'seed_predictions_h{h}.csv',seeds)
        csv(OUT/f'native_returns_h{h}.csv',native);csv(OUT/f'context_h{h}.csv',context)
        save(OUT/f'calibration_decisions_h{h}.json',decisions);csv(OUT/f'calibration_reuse_h{h}.csv',reuse)
        mature=allrows[allrows.matured];common=context[context.matured].date.tolist()
        for name,g in mature.groupby('method',sort=False):
            g=g.sort_values('date');assert g.date.tolist()==common,(h,name,len(g),len(common));assert g.date.is_unique
            metrics.append(dict(h=h,method=name,**metric(g.actual_up,g.probability)))
            for year,sg in g.groupby('year'):yearly.append(dict(h=h,method=name,year=int(year),**metric(sg.actual_up,sg.probability)))
            for state,sg in g.groupby('state'):states.append(dict(h=h,method=name,state=int(state),**metric(sg.actual_up,sg.probability)))
            bins=np.minimum((g.probability.to_numpy()*10).astype(int),9)
            for b in range(10):
                sg=g[bins==b]
                reliability.append(dict(h=h,method=name,bin_lower=b/10,bin_upper=(b+1)/10,n=len(sg),
                    mean_probability=float(sg.probability.mean()) if len(sg) else None,actual_up_frequency=float(sg.actual_up.mean()) if len(sg) else None))
            if name.startswith('cal.'):
                bg=mature[mature.method.eq(cfg()['primary'])].sort_values('date')
                updates.append(dict(h=h,method=name,adjusted_dates=int(g.adjusted.fillna(False).sum()),
                    probability_changed_dates=int((abs(g.probability.to_numpy()-bg.probability.to_numpy())>1e-12).sum()),
                    direction_changed_dates=int(((g.probability.to_numpy()>.5)!=(bg.probability.to_numpy()>.5)).sum())))
        for (name,seed),g in seeds[seeds.matured].groupby(['method','seed']):
            seedmetrics.append(dict(h=h,method=name,seed=int(seed),**metric(g.actual_up,g.probability)))
        for name,g in native[native.matured].groupby('method'):
            native_metrics.append(dict(h=h,method=name,n=len(g),mse=float(((g.predicted_return-g.actual_return)**2).mean()),
                accuracy=float(((g.predicted_return>0)==g.actual_up).mean()),predicted_sd=float(g.predicted_return.std(ddof=0)),
                actual_sd=float(g.actual_return.std(ddof=0)),correlation=float(g.predicted_return.corr(g.actual_return))))
        primary=mature[mature.method.eq(cfg()['primary'])].sort_values('date')
        for control in ['frequency_5y','market4_5y']:
            cg=mature[mature.method.eq(control)].sort_values('date')
            delta=(primary.probability.to_numpy()-primary.actual_up.to_numpy())**2-(cg.probability.to_numpy()-cg.actual_up.to_numpy())**2
            for block in [20,40]:boots.append(bootstrap_delta(delta,h,control,block))
        last=allrows[allrows.date.eq(cfg()['data_end'])]
        for row in last.to_dict('records'):
            latest.append(dict(h=h,method=row['method'],asof=row['date'],probability=float(row['probability']),
                training_cutoff=row['training_cutoff'],state=int(row['state']),matured=bool(row['matured'])))
        cm=context[context.matured];edges=np.zeros(len(data()))
        for row in cm.itertuples():edges[row.anchor+1:row.exit+1]+=1
        friday=primary[pd.to_datetime(primary.date).dt.weekday.eq(4)]
        counts.append(dict(h=h,mature_daily=len(cm),pending_daily=int((~context.matured).sum()),
            greedy_max_disjoint=maximal_nonoverlap(cm),covered_return_edges=int((edges>0).sum()),
            mean_edge_reuse=float(edges[edges>0].mean()),max_edge_reuse=int(edges.max()),
            first_signal=cm.date.min(),last_mature_signal=cm.date.max(),last_label=cm.joint_completed.max(),
            friday_only_primary=metric(friday.actual_up,friday.probability)))
    for block in [20,40]:
        family=sorted([b for b in boots if b['block']==block],key=lambda b:b['p_two_sided']);running=0.
        for i,b in enumerate(family):running=max(running,min(1.,(len(family)-i)*b['p_two_sided']));b['p_holm']=running
    for name,rows in [('metrics',metrics),('year_metrics',yearly),('state_metrics',states),('reliability',reliability),
        ('bootstrap',boots),('seed_metrics',seedmetrics),('native_metrics',native_metrics),('calibration_coverage',updates)]:csv(OUT/(name+'.csv'),pd.DataFrame(rows))
    save(OUT/'sample_counts.json',counts);save(OUT/'latest_probabilities.json',dict(computed_utc=now(),
        provenance='retrospective historical fits; current as-of model outputs, not past live issued forecasts',rows=latest))
    check_freeze();save(OUT/'scoring_completed.json',dict(status='PASS',completed_utc=now(),
        candidates_per_horizon={str(h):sum(x['h']==h for x in metrics) for h in [1,5]}))

if __name__=='__main__':main()
