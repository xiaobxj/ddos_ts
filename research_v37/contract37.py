from common37 import *
from base_contract35 import independent_route

def synthetic():
    c=cfg()['correction'];z=np.zeros(80);y=np.tile([0.,1.],40);s=np.repeat(np.arange(4),20);r=corrections(z,y,s,c)
    assert all(abs(v['offset'])<1e-15 for v in r['states']) and abs(r['shared']['offset'])<1e-15
    for target,status in [(np.ones(1000),'upper_bound'),(np.zeros(1000),'lower_bound')]:
        f=solve(np.zeros(1000),target,20.,.5);assert f['status']==status and abs(f['offset'])==.5
    r=corrections(np.zeros(39),np.ones(39),np.r_[np.zeros(19,int),np.ones(20,int)],c)
    assert r['states'][0]['offset']==0 and r['states'][0]['status']=='insufficient_new_samples';assert r['states'][1]['offset']>0 and r['eligible_states']==1 and r['shared']['offset']==r['states'][1]['offset'];assert r['states'][2]['n']==0
    r=corrections(np.array([]),np.array([]),np.array([],int),c);assert r['eligible_states']==0 and r['shared']['offset']==0
    # Shared-control penalty is exactly the sum of K individual state penalties.
    z=np.linspace(-2,2,80);y=np.r_[np.ones(20),np.zeros(20),np.ones(20),np.zeros(20)];s=np.repeat(np.arange(4),20);r=corrections(z,y,s,c);d=r['shared']['offset']
    assert abs(sum(scalar_terms(z[s==j],y[s==j],d,20.)[0] for j in range(4))-r['shared']['objective'])<1e-12
    assert sum(v['objective'] for v in r['states'])<=r['shared']['objective']+1e-12
    bound=np.tanh(.5/4);assert abs((probability(.25)-probability(-.25))-bound)<1e-15
    assert not bool(probability(0.)>.5)
    return ['zero_residual_zero_offset','both_bound_KKT','single_class_finite','19_fallback_20_fit','empty_state_and_empty_quarter','one_eligible_state_equals_shared','shared_is_equality_restricted_objective','state_training_objective_no_worse_than_shared','bounded_probability_change','strict_half_tie_down']

def causal_contract():
    obs,price,targets=data();route=csv('routing');states=csv('state_observations');raw=previous_round.descriptor_matrix(obs,price);thresholds=read(OUT/'state_thresholds.json');rows=[]
    canonical=[i for i,r in obs.iterrows() if r.weekday==4 and r.date>='2021-01-01' and isinstance(r.joint_completed,str) and r.joint_completed<=cfg()['label_end']];assert route.row_index.tolist()==canonical
    for r in route.itertuples():assert (r.encoder_cutoff,r.head_cutoff)==independent_route(r.date)
    for t in thresholds:
        a=t['encoder_cutoff'];selected=training_rows(obs,a);ordered=sorted(float(raw[i,1]) for i in selected);n=len(ordered);median=(ordered[(n-1)//2]+ordered[n//2])/2;assert median==t['volatility_median']
        for r in states[states.encoder_cutoff.eq(a)].itertuples():
            trend=raw[r.row_index,0];vol=raw[r.row_index,1];expected=('negative' if trend<0 else 'nonnegative')+('_low' if vol<=median else '_high');assert r.state==expected
    for cutoff in cfg()['decision_dates']:
        a=annual_for(cutoff)
        def selected(c):
            lo=(pd.Timestamp(c)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d');return {i for i,r in obs.iterrows() if r.date>=lo and isinstance(r.joint_completed,str) and r.joint_completed<=c}
        wanted=sorted(selected(cutoff)-selected(a));ids=new_rows(obs,cutoff);assert ids.tolist()==wanted;assert all(a<obs.joint_completed.iloc[i]<=cutoff for i in ids)
        archived=csv('membership_roles');assert wanted==archived[archived.cutoff.eq(cutoff)&archived.role.eq('added')].row_index.tolist()
        te=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);assert not len(np.intersect1d(ids,te));assert obs.date.iloc[te].gt(cutoff).all()
        s=state_ids(ids,a,states)
        for j,state in enumerate(STATES):rows.append(dict(cutoff=cutoff,encoder_cutoff=a,state=state,new_n=int((s==j).sum()),eligible=int((s==j).sum())>=cfg()['correction']['minimum_new_state_n']))
    result=pd.DataFrame(rows);assert len(result)==92;return result
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();start=now();cases=synthetic();table=causal_contract();p=OUT/'support_contract.csv';table.to_csv(p,index=False);check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=cases,state_cells=92,artifacts={str(p.relative_to(ROOT)):sha(p)}));print('Contract PASS: frozen6annual boundaries,23causal routes and bounded-offset edge cases.',flush=True)
if __name__=='__main__':main()
