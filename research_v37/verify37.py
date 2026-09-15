from common37 import *
from contract37 import synthetic,causal_contract
from verify28 import independent_inputs
from scipy.special import expit
from scipy.optimize import brentq
import math

def independently_solve(z,y,lam,cap):
    def gradient(d):return math.fsum(float(expit(float(t)+d))-int(v) for t,v in zip(z,y))+lam*d
    if gradient(-cap)>=0:return -cap
    if gradient(cap)<=0:return cap
    return float(brentq(gradient,-cap,cap,xtol=1e-14,rtol=1e-14))

def fitting_checks():
    obs,price,targets=data();mf=prior.market(price,obs,np.arange(len(obs)));u=prior.gates(price,obs,np.arange(len(obs)));y=(targets['returns']>0).astype(float);states=csv('state_observations');annuals=read(OUT/'annual_heads.json');banks=read(OUT/'feature_banks.json');heads=read(OUT/'correction_heads.json');inputs=read(OUT/'training_inputs.json');settings=cfg()['correction'];solutions=[];maxlogit=0.;maxsolution=0.;poison_cases=0
    for t in inputs:
        cutoff=t['cutoff'];annual=t['encoder_cutoff'];seed=t['seed'];rows=new_rows(obs,cutoff);d=arrays(t);ah=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED];bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));ad=arrays(ah[0]);xx=independent_inputs(bank_slice(bank,rows),mf[rows],u[rows],ad);z=np.column_stack([x@np.asarray(h['coefficients'])[:-1]+h['coefficients'][-1] for x,h in zip(xx,ah)])
        np.testing.assert_array_equal(d['row_index'],rows);np.testing.assert_array_equal(d['direction'],y[rows]);np.testing.assert_array_equal(d['state_index'],state_ids(rows,annual,states));gap=float(abs(z-d['annual_logits']).max());assert gap<1e-10;maxlogit=max(maxlogit,gap)
        # R25's base coefficients and interaction gamma remain precisely annual.
        t19=np.asarray(ah[1]['coefficients']);t25=np.asarray(ah[3]['coefficients']);np.testing.assert_array_equal(np.r_[t25[:-2],t25[-1]],t19)
        # Poison every row that the correction interface is not authorized to receive.
        outside=np.ones(len(obs),bool);outside[rows]=False;yp=y.copy();yp[outside]=1-yp[outside];mp=mf.copy();mp[outside]=1e9;up=u.copy();up[outside]=-1e9;bp={k:v.copy() for k,v in bank.items()};bp['features'][~np.isin(bp['row_index'],rows)]=1e9
        isolated=training_interface(rows,annual,bp,mp,up,yp,states,ah)
        for key in d:np.testing.assert_array_equal(d[key],isolated[key])
        poison_cases+=1
        for j,m in enumerate(LEARNED):
            h=next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m);si=d['state_index'];count=np.bincount(si,minlength=4);eligible=count>=20;assert h['eligible_states']==int(eligible.sum()) and h['new_n']==len(rows)
            for k,item in enumerate(h['states']):
                mask=si==k;assert item['state']==STATES[k] and item['n']==int(mask.sum()) and item['up_n']==int(y[rows][mask].sum()) and item['eligible']==bool(eligible[k])
                if not eligible[k]:assert item['offset']==0.;continue
                alt=independently_solve(z[mask,j],y[rows][mask],20.,.5);error=abs(alt-item['offset']);assert error<1e-10;maxsolution=max(maxsolution,error)
                q=z[mask,j]+item['offset'];objective=math.fsum(float(np.logaddexp(0.,-v if label else v)) for v,label in zip(q,y[rows][mask]))+10*item['offset']**2;assert abs(objective-item['objective'])<1e-9;assert item['kkt_violation']<1e-10
                solutions.append(dict(cutoff=cutoff,seed=seed,method=m,state=STATES[k],kind='state',coefficient_gap=error))
            mask=eligible[si];item=h['shared'];assert item['n']==int(mask.sum()) and item['lambda_sum']==20*int(eligible.sum())
            if eligible.any():
                alt=independently_solve(z[mask,j],y[rows][mask],20*int(eligible.sum()),.5);error=abs(alt-item['offset']);assert error<1e-10;maxsolution=max(maxsolution,error);solutions.append(dict(cutoff=cutoff,seed=seed,method=m,state='eligible_shared',kind='global',coefficient_gap=error))
                assert sum(scalar_terms(z[si==k,j],y[rows][si==k],h['states'][k]['offset'],20.)[0] for k in np.flatnonzero(eligible))<=item['objective']+1e-9
            else:assert item['offset']==0.
    for h in heads:
        if h['cutoff']==h['encoder_cutoff']:assert h['new_n']==0 and h['eligible_states']==0 and h['shared']['offset']==0 and all(s['offset']==0 for s in h['states'])
    flat=[]
    for h in heads:
        key={k:h[k] for k in ['cutoff','encoder_cutoff','seed','method']};flat.extend(dict(**key,kind='state',**item) for item in h['states']);flat.append(dict(**key,kind='global',state='eligible_shared',eligible=h['eligible_states']>0,**h['shared']))
    pd.testing.assert_frame_equal(pd.DataFrame(flat),csv('correction_parameters'),check_dtype=False,atol=1e-14,rtol=0)
    return pd.DataFrame(solutions),dict(maximum_training_logit_gap=maxlogit,maximum_scalar_solution_gap=maxsolution,independent_scalar_solutions=len(solutions),outside_member_poison_interfaces=poison_cases)

def independent_metric(g):
    n=len(g);y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool);q=g.probability.to_numpy(float);score=g.score.to_numpy(float);positive=int(y.sum());negative=n-positive
    r=dict(n=n,correct_directions=int((y==up).sum()),tp=int((y&up).sum()),tn=int((~y&~up).sum()),fp=int((~y&up).sum()),fn=int((y&~up).sum()))
    r.update(accuracy=r['correct_directions']/n if n else None,direction_error=(n-r['correct_directions'])/n if n else None,balanced_accuracy=(r['tp']/positive+r['tn']/negative)/2 if positive and negative else None,predicted_up_fraction=int(up.sum())/n if n else None,observed_up_fraction=positive/n if n else None,auroc=None,brier=None,log_loss=None,mean_probability=None,probability_std=None,calibration_gap=None,clipped_probabilities=None)
    if positive and negative:r['auroc']=math.fsum(float(a>b)+.5*float(a==b) for a in score[y] for b in score[~y])/(positive*negative)
    if n and np.isfinite(q).all():
        mean=math.fsum(q)/n;bounded=np.clip(q,1e-12,1-1e-12);r.update(brier=math.fsum((float(p)-int(v))**2 for p,v in zip(q,y))/n,log_loss=-math.fsum(math.log(float(p)) if v else math.log1p(-float(p)) for p,v in zip(bounded,y))/n,mean_probability=mean,probability_std=math.sqrt(math.fsum((float(p)-mean)**2 for p in q)/n),calibration_gap=mean-positive/n,clipped_probabilities=int(((q<1e-12)|(q>1-1e-12)).sum()))
    return r
def assert_metric(g,row):
    for k,v in independent_metric(g).items():
        if v is None:assert pd.isna(row[k]),(k,row[k])
        else:assert abs(v-row[k])<1e-12,(k,v,row[k])

def prediction_checks():
    obs,price,targets=data();models=csv('model_predictions');ensemble=csv('ensemble_predictions');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');details=csv('seed_correction_effects');route=csv('routing');states=csv('state_observations');annuals=read(OUT/'annual_heads.json');banks=read(OUT/'feature_banks.json');heads=read(OUT/'correction_heads.json');maxgap=0.;temporal=[]
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    di=details.set_index(['history','method','seed','row_index']);mi=models.set_index(['history','method','seed','row_index'])
    for cutoff,g in route.groupby('head_cutoff',sort=True):
        annual=annual_for(cutoff);rows=g.row_index.to_numpy(int);s=state_ids(rows,annual,states);anchor=int(obs.anchor.iloc[rows].max());poison=price.copy();mask=np.arange(len(price))>anchor;poison.loc[mask,['open','high','low','close','volume']]*=7
        np.testing.assert_array_equal(prior.market(price,obs,rows),prior.market(poison,obs,rows));np.testing.assert_array_equal(prior.gates(price,obs,rows),prior.gates(poison,obs,rows));temporal.append(dict(cutoff=cutoff,weekly_n=len(rows),future_price_invariance=True))
        for seed in cfg()['seeds']:
            ah=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED];bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));xx=independent_inputs(bank_slice(bank,rows),prior.market(price,obs,rows),prior.gates(price,obs,rows),arrays(ah[0]))
            for m,x,h0 in zip(LEARNED,xx,ah):
                z=x@np.asarray(h0['coefficients'])[:-1]+h0['coefficients'][-1];h=next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m)
                for kind,history in POLICIES.items():
                    for i,idx in enumerate(rows):
                        d=di.loc[(history,m,seed,idx)];r=mi.loc[(history,m,seed,idx)];b=mi.loc[(ANNUAL,m,seed,idx)];item=h['states'][s[i]];off=0. if not item['eligible'] else h['shared']['offset'] if kind=='global' else item['offset']*(.5 if kind=='half' else 1.)
                        expected=b.probability if off==0 else float(expit(z[i]+off));gap=abs(expected-r.probability);maxgap=max(maxgap,gap);assert gap<1e-12
                        assert d.logit_offset==off and abs(d.annual_logit-z[i])<1e-10 and d.annual_probability==b.probability and d.probability==r.probability and d.probability_change==r.probability-b.probability;assert d.state==STATES[s[i]] and d.new_state_n==item['n'] and d.eligible==item['eligible']
                        assert abs(r.probability-b.probability)<=np.tanh((.25 if kind=='half' else .5)/4)+1e-12
                        assert r.direction_up==int(r.probability>.5) and r.actual_up==int(targets['returns'][idx]>0) and abs(r.actual-targets['returns'][idx])<1e-14 and r.cutoff==cutoff
                        if not item['eligible']:assert r.probability==b.probability
    for history in NEW:
        a=models[models.history.eq(history)&~models.method.isin(LEARNED)].drop(columns='history').reset_index(drop=True);b=base[base.history.eq(ANNUAL)&~base.method.isin(LEARNED)].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
    grouped=models.groupby(['history','method','date'],sort=False);ei=ensemble.set_index(['history','method','date'])
    for key,g in grouped:
        r=ei.loc[key];native=key[1]=='native_mse';n=1 if key[1]=='training_frequency' else 3;assert len(g)==n;score=math.fsum(g.score)/n;assert abs(r.score-score)<1e-14
        if native:assert pd.isna(r.probability);assert r.direction_up==int(score>0)
        else:assert abs(r.probability-math.fsum(g.probability)/n)<1e-14;assert r.direction_up==int(r.probability>.5)
    for table,source,keys in [('ensemble_metrics',ensemble,['history','method']),('seed_metrics',models,['history','method','seed']),('state_metrics',ensemble.merge(csv('weekly_context')[['row_index','date','state']],on=['row_index','date'],validate='many_to_one'),['history','method','state'])]:
        by={k:v for k,v in source.groupby(keys,sort=False)}
        for row in csv(table).to_dict('records'):
            g=by[tuple(row[k] for k in keys)] if tuple(row[k] for k in keys) in by else source.iloc[:0];w=next(w for w in periods() if w['name']==row['period']);g=g[g.date.between(w['start'],w['end'])];assert_metric(g,row)
    for r in csv('weekly_correction_effects').itertuples():
        a=ei.loc[(r.history,r.method,r.date)];b=ei.loc[(ANNUAL,r.method,r.date)];assert a.probability==r.probability and b.probability==r.annual_probability;good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;case='regression' if old and not good else 'recovery' if good and not old else 'stable_correct' if old else 'stable_wrong';assert r.case==case and r.correct==good and r.annual_correct==old
        assert r.probability_change==a.probability-b.probability and r.signed_probability_change==(2*a.actual_up-1)*r.probability_change
    for r in csv('direction_changes').itertuples():
        w=next(w for w in periods() if w['name']==r.period);g=csv('weekly_correction_effects');g=g[g.history.eq(r.history)&g.method.eq(r.method)&g.date.between(w['start'],w['end'])];assert len(g)==r.n and int(g.eligible.sum())==r.eligible_weeks and r.fallback_weeks==r.n-r.eligible_weeks;assert int(g['case'].eq('recovery').sum())==r.recoveries and int(g['case'].eq('regression').sum())==r.regressions;assert abs(g.signed_probability_change.mean()-r.mean_signed_probability_change)<1e-14
    return pd.DataFrame(temporal),dict(maximum_prediction_gap=maxgap,independent_ensemble_rows=len(ensemble),independent_metric_cells=len(csv('ensemble_metrics'))+len(csv('seed_metrics'))+len(csv('state_metrics')),exact_old_model_rows=len(base),exact_fallback_seed_rows=int((~details.eligible).sum()))

def inference_checks():
    e=csv('ensemble_predictions');pairs=read(OUT/'primary_comparisons.json');ps=[]
    for w in cfg()['windows']:
        n=w['n'];starts=np.random.default_rng(20260910).integers(n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);np.testing.assert_array_equal(ids,statistics.bootstrap_indices(n));part=e[e.date.between(w['start'],w['end'])]
        for row in [p for p in pairs if p['window']==w['name']]:
            a=part[part.history.eq(row['history'])&part.method.eq(row['candidate'])].sort_values('date');b=part[part.history.eq(row['reference_history'])&part.method.eq(row['reference'])].sort_values('date');assert len(a)==n and a.date.tolist()==b.date.tolist()
            values=(a.direction_up.to_numpy()!=a.actual_up.to_numpy()).astype(float)-(b.direction_up.to_numpy()!=b.actual_up.to_numpy()).astype(float) if row['metric']=='direction_error' else (a.probability.to_numpy()-a.actual_up.to_numpy())**2-(b.probability.to_numpy()-b.actual_up.to_numpy())**2
            mean=float(values.mean());boot=values[ids].mean(1);center=(values-mean)[ids].mean(1);p=(1+int((abs(center)>=abs(mean)).sum()))/10001;lo,hi=np.quantile(boot,[.025,.975]);assert abs(mean-row['difference'])<1e-14 and abs(lo-row['ci95_low'])<1e-14 and abs(hi-row['ci95_high'])<1e-14 and p==row['p'];ps.append(p)
    adjusted=[0.]*len(ps);last=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):last=max(last,min(1.,ps[i]*(len(ps)-rank)));adjusted[i]=last
    np.testing.assert_array_equal(adjusted,[p['holm_adjusted_p'] for p in pairs]);assert len(pairs)==48
def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();start=time.time();last=read(OUT/'contract_verification.json')['completed_utc']
    for phase in ['fitting','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    synthetic();pd.testing.assert_frame_equal(causal_contract(),csv('support_contract'),check_dtype=False,check_exact=True)
    sol,fit=fitting_checks();print('Independent inputs, scalar solvers and51member isolation checks PASS.',flush=True);temporal,pred=prediction_checks();print('Independent predictions, metrics and future-price checks PASS.',flush=True);inference_checks()
    assert fit['independent_scalar_solutions']==read(OUT/'fitting_manifest.json')['new_scalar_fits'];assert old_evidence()==prep['old_evidence'];files=[]
    for name,g in [('independent_solutions',sol),('temporal_checks',temporal)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],independent_comparisons=48,old_files_preserved=5315,**fit,**pred,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('R37 independent verification PASS.',flush=True)
if __name__=='__main__':main()
