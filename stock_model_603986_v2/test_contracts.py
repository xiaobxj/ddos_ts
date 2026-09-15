"""Deterministic checks on adaptation decisions before historical scoring."""
from core import *
from calibration import decision,split_bank,accepted
sys.path.insert(0,str(ROOT))  # Legacy adapters add v1 paths; select this experiment's scorer.
from score import metric

def main():
    checks=[]
    dates=pd.bdate_range('2021-01-01',periods=700).strftime('%Y-%m-%d').to_numpy()
    for h in [1,5]:
        n=len(dates)-h;anchor=np.arange(n)
        bank=pd.DataFrame(dict(h=h,anchor=anchor,exit=anchor+h,date=dates[:n],joint_completed=dates[h:],
            matured=True,actual_up=1.,actual_return=.01,probability=.6,state=anchor%4))
        cutoff='2023-06-30';p,rows=decision(bank,cutoff,'quarter_four_state')
        assert all(r['accepted'] for r in rows)
        assert all(v[1]>0 for v in p.values());checks.append(f'H{h} positive held-out alignment passes')
        tr,va,ready=split_bank(bank,cutoff);assert ready and tr.joint_completed.max()<va.date.min()
        changed=bank.copy();changed.loc[changed.date.isin(va.date),'actual_up']=0
        p,rows=decision(changed,cutoff,'quarter_four_state');assert not p and all(r['reason']=='validation_quality_rejected' for r in rows)
        checks.append(f'H{h} opposite validation outcomes reject')
        mutated=bank.copy();late=mutated.joint_completed.gt(cutoff)
        mutated.loc[late,['actual_up','actual_return','probability']]=[0.,-99.,.001]
        assert decision(bank,cutoff,'quarter_four_state')==decision(mutated,cutoff,'quarter_four_state')
        checks.append(f'H{h} future endpoint mutations have no effect')
        p,rows=decision(bank,'2022-12-31','quarter_four_state');assert not p and all(r['reason']=='annual_reset' for r in rows)
        checks.append(f'H{h} Q1 reset enforced')
        sparse=bank.copy();sparse['state']=0;sparse.loc[sparse.index[-2:],'state']=3
        p,rows=decision(sparse,cutoff,'quarter_four_state');assert 3 not in p
        checks.append(f'H{h} unsupported states fall back')
    assert not accepted(np.array([0,1]),np.array([.5,.5]),np.array([.5,.5]))[0]
    assert metric([0,1],[.5,.5])['predicted_up_fraction']==0
    assert abs(metric([0,1],[.5,.5])['brier']-.25)<1e-15
    checks.extend(['Brier tie does not pass','Strict probability threshold .5','Constant reference Brier .25'])
    f=data()
    for h in [1,5]:
        obs,target=observations(f,h);ids,_=membership(obs,'2025-12-31',5)
        r,_,_=modules();labels,scale=r.scales_for(target,ids)
        raw=target['returns'][ids]>0;wrong=labels['returns']>0
        assert np.any(raw!=wrong)
        checks.append(f'H{h} raw-up vs standardized-positive mismatch detected in {int((raw!=wrong).sum())} training rows')
    save(OUT/'synthetic_contract_checks.json',dict(status='PASS',completed_utc=now(),checks=checks,
        synthetic_only='Business-day fixture is not an exchange calendar or forecast record'))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
