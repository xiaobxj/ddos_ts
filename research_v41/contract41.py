from common41 import *
from scipy.special import expit

def synthetic():
    assert cell_gate('ready',10,5,-.01,0)==(True,'accepted')
    assert cell_gate('ready',10,4,-.01,0)==(False,'insufficient_validation')
    assert cell_gate('ready',10,5,0.,0)==(False,'brier_not_better')
    assert cell_gate('ready',10,5,-.01,-1)==(False,'direction_worse')
    assert cell_gate('q1_guard',30,13,-.1,3)==(False,'q1_guard')
    # A five-week positive margin remains positive after one deletion; only count can reject.
    d=np.full(5,-.01);assert ((d.sum()-d)/4<0).all()
    # Six-week net gain depends entirely on one helpful observation.
    d=np.array([-.11,.02,.02,.02,.02,.02]);assert d.mean()<0 and (d.sum()-d[0])/5>0
    previous={1,2,3,4,5,6,7};current={5,6,7};assert len(current)==len(previous)-len(previous-current)+len(current-previous)
    assert len(previous&current)==3 and len(previous-current)==4
    return ['original_gate_boundaries','mechanical_count_vs_metric_sensitivity','single_helpful_observation','support_flow_identity']

def build():
    bank=csv('weekly_signal_bank');bi=bank.set_index(['method','seed','row_index']);state=bank[['row_index','state']].drop_duplicates().set_index('row_index').state
    obs,_,target=data();cells=[];effects=[];maturity_checks=0
    for cadence in CADENCES:
        gates=csv(cadence+'_gates');gates=gates[gates.family.eq('state')];schedule=csv(cadence+'_schedule').set_index('cutoff');members=csv(cadence+'_members');vp=csv(cadence+'_validation_predictions');vp=vp[vp.family.eq('state')]
        heads=read(OUT/f'{cadence}_heads.json');hi={(r['cutoff'],r['method'],r['seed'],r['component']):r for r in heads if r['family']=='state'}
        assert not vp.duplicated(['cutoff','method','component','row_index','seed']).any()
        for cutoff,c in schedule.iterrows():
            tr=members[members.cutoff.eq(cutoff)&members.role.eq('training')];va=members[members.cutoff.eq(cutoff)&members.role.eq('validation')]
            assert len(tr)==c.train_n and len(va)==c.validation_n and not set(tr.row_index)&set(va.row_index)
            if len(tr):assert tr.joint_completed.max()<va.date.min() and tr.joint_completed.max()<=c.fit_cutoff
            if len(va):assert va.joint_completed.le(cutoff).all()
            for role in [tr,va]:
                for r in role.itertuples():assert obs.iloc[r.row_index].date==r.date and obs.iloc[r.row_index].joint_completed==r.joint_completed
            maturity_checks+=1
        for (cutoff,method,st,idx),g in vp.groupby(['cutoff','method','component','row_index'],sort=True):
            g=g.sort_values('seed');assert g.seed.tolist()==cfg()['seeds'];r=g.iloc[0];assert r.date<r.joint_completed<=cutoff and state.loc[idx]==st
            for t in g.itertuples():
                b=bi.loc[(method,t.seed,idx)];h=hi[(cutoff,method,t.seed,st)];assert t.annual_probability==b.probability and t.offset==h['offset'] and t.actual_up==int(target['returns'][idx]>0)
                expected=b.probability if t.offset==0 else float(expit(b.annual_logit+t.offset));eq(t.candidate_probability,expected)
            p0=float(g.annual_probability.mean());p=float(g.candidate_probability.mean());y=int(r.actual_up);bc=int((p0>.5)==bool(y));cc=int((p>.5)==bool(y))
            effects.append(dict(cadence=cadence,cutoff=cutoff,method=method,state=st,row_index=int(idx),date=r.date,joint_completed=r.joint_completed,actual_up=y,annual_probability=p0,candidate_probability=p,baseline_loss=(p0-y)**2,candidate_loss=(p-y)**2,brier_difference=(p-y)**2-(p0-y)**2,baseline_correct=bc,candidate_correct=cc,correct_difference=cc-bc))
        e=pd.DataFrame([r for r in effects if r['cadence']==cadence]);eg={k:g for k,g in e.groupby(['cutoff','method','state'])}
        for r in gates.to_dict('records'):
            cutoff=r['cutoff'];st=r['component'];method=r['method'];g=eg.get((cutoff,method,st),e.iloc[:0]);n=len(g);tr=members[members.cutoff.eq(cutoff)&members.role.eq('training')];va=members[members.cutoff.eq(cutoff)&members.role.eq('validation')];ids=va[va.row_index.map(state).eq(st)].row_index.tolist()
            assert sorted(g.row_index.tolist())==sorted(ids);assert r['training_n']==int(tr.row_index.map(state).eq(st).sum())
            bd=float(g.brier_difference.mean()) if n else None;cd=int(g.correct_difference.sum());accepted,reason=cell_gate(r['mode'],r['training_n'],n,bd,cd)
            assert r['accepted']==accepted and r['reason']==reason and r['validation_n']==n and r['baseline_correct']==int(g.baseline_correct.sum()) and r['candidate_correct']==int(g.candidate_correct.sum());eq(r['brier_difference'],bd)
            cells.append(dict(cadence=cadence,cutoff=cutoff,method=method,state=st,mode=r['mode'],training_n=r['training_n'],validation_n=n,accepted=accepted,reason=reason,brier_difference=bd,correct_difference=cd,baseline_correct=r['baseline_correct'],candidate_correct=r['candidate_correct'],metric_eligible=r['mode']=='ready' and r['training_n']>=10 and n>=5))
    assert len(cells)==1456 and maturity_checks==91
    return pd.DataFrame(cells),pd.DataFrame(effects)

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();start=now();tests=synthetic();cells,effects=build();files=[]
    for name,g in [('gate_cells',cells),('validation_week_effects',effects)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=tests,gate_cells=len(cells),validation_week_cells=len(effects),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print(f'Contract PASS:1456archived gates and {len(effects)}validation-week contributions.',flush=True)
if __name__=='__main__':main()
