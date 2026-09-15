from common38 import *
from base_contract35 import independent_route
def synthetic():
    g=pd.DataFrame(dict(entry=[0,2,5],exit=[5,7,10]));r=overlap(g);assert r==dict(n=3,unique_intervals=3,total_edge_uses=15,union_edges=10,mean_edge_multiplicity=1.5,maximum_edge_multiplicity=2,overlapping_pairs=2,maximum_disjoint_intervals=2)
    r=overlap(pd.DataFrame(dict(entry=[0,5],exit=[5,10])));assert r['overlapping_pairs']==0 and r['maximum_disjoint_intervals']==2 and r['mean_edge_multiplicity']==1
    assert overlap(g.iloc[:0])['mean_edge_multiplicity'] is None
    a=decomposition([10,10,0,0],[2,-1,0,0],[5,15,0,0],[.5,-3,0,0]);assert a['status']=='defined' and abs(a['composition']+a['within_state']+.175)<1e-12
    assert decomposition([1,0,0,0],[.1,0,0,0],[0,1,0,0],[0,.2,0,0])['status']=='target_state_absent_in_source'
    assert decomposition([0]*4,[0]*4,[1]*4,[0]*4)['status']=='empty_source'
    assert agreement(.1,-.1)=='opposite' and agreement(0,.1)=='near_zero' and agreement(None,.1)=='undefined'
    p=np.array([.2,.5,.8]);q=np.array([.3,.5,.7]);y=np.array([1.,0.,0.]);brier_terms(p,q,y);assert ranking(p,q)['inversions']==0 and ranking(p,q)['tie_changes']==0
    assert ranking([.2,.5],[.5,.2])['inversions']==1 and ranking([.5,.5],[.4,.5])['tie_changes']==1
    return ['overlap_edges_and_pairs','touching_boundaries_do_not_overlap','empty_overlap','signed_mean_decomposition_identity','unseen_state_not_imputed','empty_source','residual_sign_ties_and_missing','Brier_identity','monotone_order_and_tie_changes']
def memberships():
    obs,price,targets=data();route=csv('routing');states=csv('state_observations');members=[];checks=[]
    for r in route.itertuples():assert (r.encoder_cutoff,r.head_cutoff)==independent_route(r.date)
    assert len(route)==272 and route.row_index.is_unique
    for cutoff in cfg()['decision_dates']:
        annual=previous_round.annual_for(cutoff)
        def select(c):
            lo=(pd.Timestamp(c)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d');return {i for i,r in obs.iterrows() if r.date>=lo and isinstance(r.joint_completed,str) and r.joint_completed<=c}
        daily=np.array(sorted(select(cutoff)-select(annual)),int);np.testing.assert_array_equal(daily,previous_round.new_rows(obs,cutoff));friday=daily[obs.weekday.iloc[daily].to_numpy()==4];future=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int)
        assert all(annual<obs.joint_completed.iloc[i]<=cutoff for i in daily);assert all(obs.date.iloc[i]>cutoff and obs.joint_completed.iloc[i]<=cfg()['label_end'] for i in future);assert not set(daily)&set(future)
        if len(daily):assert int(obs.exit.iloc[daily].max())<=int(obs.entry.iloc[future].min())
        for view,ids in [('daily',daily),('friday',friday),('future',future)]:
            ss=previous_round.state_ids(ids,annual,states)
            for i,s in zip(ids,ss):
                r=obs.iloc[i];assert int(r.entry)>int(r.anchor) and int(r.exit)>int(r.entry);assert r.joint_completed>=price.date.iloc[int(r.exit)]
                members.append(dict(cutoff=cutoff,encoder_cutoff=annual,year=int(annual[:4])+1,view=view,row_index=int(i),date=r.date,weekday=int(r.weekday),state=STATES[s],entry=int(r.entry),exit=int(r.exit),joint_completed=r.joint_completed))
        for seed in cfg()['seeds']:
            refs=[r for r in read(OUT/'training_inputs.json') if r['cutoff']==cutoff and r['seed']==seed];assert len(refs)==int(len(daily)>0)
            if refs:
                d=arrays(refs[0]);np.testing.assert_array_equal(d['row_index'],daily);np.testing.assert_array_equal(d['direction'],(targets['returns'][daily]>0).astype(float));np.testing.assert_array_equal(d['state_index'],previous_round.state_ids(daily,annual,states));assert d['annual_logits'].shape==(len(daily),4)
        checks.append(dict(cutoff=cutoff,daily_n=len(daily),friday_n=len(friday),future_n=len(future),training_mature=True,train_future_execution_disjoint=True))
    return pd.DataFrame(members),pd.DataFrame(checks)
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();start=now();cases=synthetic();members,checks=memberships();rows=[]
    for cutoff in cfg()['decision_dates']:
        for view in VIEWS:
            g=members[members.cutoff.eq(cutoff)&members.view.eq(view)]
            for state in SCOPES:
                s=g if state=='all' else g[g.state.eq(state)];rows.append(dict(cutoff=cutoff,year=int(previous_round.annual_for(cutoff)[:4])+1,view=view,state=state,**overlap(s)))
    overlaps=pd.DataFrame(rows);assert len(overlaps)==345;files=[]
    for name,g in [('membership',members),('membership_checks',checks),('target_overlap',overlaps)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=cases,label_conditioned_summaries=False,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Contract PASS; memberships and345label-free overlap cells frozen.',flush=True)
if __name__=='__main__':main()
