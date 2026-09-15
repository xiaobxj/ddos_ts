"""Causal, bounded scalar corrections on an immutable annual neural classifier."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V36=PROJECT/'research_v36'
sys.path.insert(0,str(V36));import common36 as previous_round
prior=previous_round.prior;v35=previous_round.previous_round
np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data;arrays=prior.arrays;ref=prior.ref;metric=prior.metric
probability=prior.probability;design=prior.design;apply_pipeline=prior.apply_pipeline;statistics=prior.statistics
STATES=previous_round.STATES;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY;ARMS=v35.ARMS;ANNUAL=ARMS['annual']
POLICIES={'state':'annual_state_offset','global':'annual_shared_offset','half':'annual_state_offset_half'};NEW=list(POLICIES.values())
annual_for=v35.annual_for;bank_slice=v35.bank_slice;training_rows=v35.training_rows;ensemble_from=v35.ensemble_from

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V36/'protocol.json']+[V36/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md','annual_heads.json','feature_banks.json','routing.csv','state_thresholds.json','state_observations.csv','membership_roles.csv','training_state_summary.csv','weekly_context.csv','model_predictions.csv','ensemble_predictions.csv']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V36/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V36/'results/delivery_manifest.json')['files'].items():r[str((V36/n).relative_to(PROJECT))]=d
    p=V36/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5315
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==p['source_sha256'] and c['protocol_sha256']==p['protocol_sha256']
        for n,d in c['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def periods():return cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end=cfg()['label_end'],n=272)]+[dict(name=f'year_{y}',start=f'{y}-01-01',end=f'{y}-12-31') for y in range(2021,2027)]
def new_rows(obs,cutoff):return np.setdiff1d(training_rows(obs,cutoff),training_rows(obs,annual_for(cutoff)))
def state_ids(rows,annual,states):
    s=states[states.encoder_cutoff.eq(annual)].set_index('row_index').loc[rows,'state'];return np.array([STATES.index(v) for v in s],int)
def scalar_terms(z,y,d,lam):
    q=np.asarray(z,float)+d;p=probability(q)
    return float(np.where(y>0,np.logaddexp(0.,-q),np.logaddexp(0.,q)).sum()+lam*d*d/2),float((p-y).sum()+lam*d),float((p*(1-p)).sum()+lam)
def solve(z,y,lam,cap,iterations=60):
    z=np.asarray(z,float);y=np.asarray(y,float);assert len(z)>0 and z.shape==y.shape and np.isfinite(z).all() and np.isin(y,[0.,1.]).all() and lam>0 and cap>0
    lo=-cap;hi=cap;glo=scalar_terms(z,y,lo,lam)[1];ghi=scalar_terms(z,y,hi,lam)[1]
    if glo>=0:d=lo;status='lower_bound'
    elif ghi<=0:d=hi;status='upper_bound'
    else:
        for _ in range(iterations):
            mid=(lo+hi)/2
            if scalar_terms(z,y,mid,lam)[1]>0:hi=mid
            else:lo=mid
        d=(lo+hi)/2;status='interior'
    value,g,h=scalar_terms(z,y,d,lam);violation=abs(g) if status=='interior' else max(0.,-g) if status=='lower_bound' else max(0.,g)
    assert violation<1e-10 and value<=scalar_terms(z,y,0.,lam)[0]+1e-10
    return dict(offset=float(d),status=status,objective=value,gradient=g,hessian=h,kkt_violation=violation,lambda_sum=float(lam),n=len(y),up_n=int(y.sum()),up_rate=float(y.mean()),annual_mean_probability=float(probability(z).mean()),annual_log_loss=scalar_terms(z,y,0.,0.)[0]/len(y),corrected_log_loss=(value-lam*d*d/2)/len(y))
def corrections(z,y,s,settings):
    z=np.asarray(z,float);y=np.asarray(y,float);s=np.asarray(s,int);assert z.shape==y.shape==s.shape
    count=np.bincount(s,minlength=4);eligible=count>=settings['minimum_new_state_n'];k=int(eligible.sum());items=[]
    for j in range(4):
        mask=s==j;n=int(mask.sum())
        if eligible[j]:r=solve(z[mask],y[mask],settings['ridge_sum_lambda'],settings['absolute_logit_cap'],settings['bisection_iterations'])
        else:r=dict(offset=0.,status='insufficient_new_samples',lambda_sum=settings['ridge_sum_lambda'],n=n,up_n=int(y[mask].sum()),up_rate=float(y[mask].mean()) if n else None,annual_mean_probability=float(probability(z[mask]).mean()) if n else None)
        items.append(dict(state=STATES[j],eligible=bool(eligible[j]),**r))
    mask=eligible[s]
    shared=solve(z[mask],y[mask],settings['ridge_sum_lambda']*k,settings['absolute_logit_cap'],settings['bisection_iterations']) if k else dict(offset=0.,status='no_eligible_states',lambda_sum=0.,n=0,up_n=0,up_rate=None,annual_mean_probability=None)
    return dict(states=items,shared=shared,eligible_states=k)
def training_interface(rows,annual,bank,mf,u,y,states,heads):
    # Slice to the causal, newly mature members before any correction estimator sees data.
    xx=apply_pipeline(bank_slice(bank,rows),mf[rows],u[rows],arrays(heads[0]));z=np.column_stack([design(x)@np.asarray(h['coefficients']) for x,h in zip(xx,heads)])
    return dict(row_index=rows,direction=y[rows],state_index=state_ids(rows,annual,states),annual_logits=z)
def get_delta(h,kind,state_index):
    offsets=np.array([r['offset'] for r in h['states']]);eligible=np.array([r['eligible'] for r in h['states']])
    if kind=='global':offsets=np.where(eligible,h['shared']['offset'],0.)
    elif kind=='half':offsets=offsets*cfg()['correction']['half_fraction']
    return offsets[state_index]
