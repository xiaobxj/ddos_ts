"""Adaptation from previously issued causal annual OOS forecasts only."""
from core import *
from scipy.special import expit, logit
from scipy.optimize import brentq

POLICIES=['quarter_four_state','month_four_state','quarter_pooled','quarter_ungated','quarter_platt']

def split_bank(bank,cutoff):
    b=bank[bank.matured&bank.joint_completed.le(cutoff)&bank.date.le(cutoff)].sort_values('date')
    va=b.tail(63)
    if len(va)<63:return b.iloc[:0],va,False
    tr=b[b.joint_completed.lt(va.date.iloc[0])].tail(252)
    return tr,va,len(tr)==252

def offset_fit(tr):
    z=logit(np.clip(tr.probability.to_numpy(),1e-12,1-1e-12));y=tr.actual_up.to_numpy()
    f=lambda delta:float((expit(z+delta)-y).sum()+20*delta)
    if f(-.5)>=0:return -.5
    if f(.5)<=0:return .5
    return float(brentq(f,-.5,.5,xtol=1e-14))

def accepted(y,p,base):
    diff=float(np.mean((p-y)**2-(base-y)**2))
    wins=int(((p>.5)==y).sum()-((base>.5)==y).sum())
    return diff<-1e-12 and wins>=0,diff,wins

def decision(bank,cutoff,policy):
    tr,va,ready=split_bank(bank,cutoff);r,_,_=modules()
    rows=[];params={};pooled=policy in ['quarter_pooled','quarter_platt']
    reset=int(cutoff[5:7]) in ([12,1,2] if policy.startswith('month') else [12])
    for state in ([-1] if pooled else range(4)):
        t=tr if pooled else tr[tr.state.eq(state)];v=va if pooled else va[va.state.eq(state)]
        nt=maximal_nonoverlap(t);nv=maximal_nonoverlap(v)
        reason='annual_reset' if reset else ('insufficient_overall_history' if not ready else 'insufficient_train_support')
        delta=0.;slope=1.;keep=False;diff=None;wins=None;fragile=0;deletions=0
        if not reset and ready and nt>=10:
            if policy=='quarter_platt':
                # Fixed OOS logit slope fit, not an in-sample neural residual correction.
                z=logit(np.clip(t.probability.to_numpy(),1e-12,1-1e-12))
                theta,_=r.fit_newton(z[:,None],t.actual_up.to_numpy(),r.cfg()['probe']);slope,delta=map(float,theta)
            else:delta=offset_fit(t)
            if len(v):
                z=logit(np.clip(v.probability.to_numpy(),1e-12,1-1e-12));p=expit(slope*z+delta)
                keep,diff,wins=accepted(v.actual_up.to_numpy(),p,v.probability.to_numpy())
            if policy=='quarter_ungated':keep=True;reason='train_only_no_validation_gate'
            elif nv<5:keep=False;reason='insufficient_validation_support'
            else:
                reason='accepted' if keep else 'validation_quality_rejected'
                # Remove a fixed H-session anchor block; holds fitted parameters fixed.
                # This is sensitivity, not an extra post-result gate or independent trial.
                blocks=v.anchor.to_numpy()//int(bank.h.iloc[0])
                for block in np.unique(blocks):
                    mask=blocks!=block
                    if not mask.any():continue
                    ok,_,_=accepted(v.actual_up.to_numpy()[mask],p[mask],v.probability.to_numpy()[mask])
                    fragile+=int(ok!=keep);deletions+=1
        if keep:params[int(state)]=(slope,delta)
        rows.append(dict(cutoff=cutoff,policy=policy,state=state,ready=ready,reason=reason,accepted=bool(keep),
            train_n=len(t),validation_n=len(v),train_disjoint=nt,validation_disjoint=nv,delta=delta,slope=slope,
            validation_brier_delta=diff,validation_correct_delta=wins,
            delete_block_metric_flips=fragile,delete_block_checks=deletions,
            train_first=t.date.min() if len(t) else '',train_last_maturity=t.joint_completed.max() if len(t) else '',
            validation_first=v.date.min() if len(v) else '',validation_last_maturity=v.joint_completed.max() if len(v) else '',
            training_dates=t.date.tolist(),validation_dates=v.date.tolist()))
    return params,rows

def calibrate(bank):
    outputs=[];decisions=[];reuse=[]
    for policy in POLICIES:
        freq='ME' if policy.startswith('month') else 'QE'
        ends=pd.date_range('2022-12-31',cfg()['data_end'],freq=freq).strftime('%Y-%m-%d').tolist()
        result=bank.copy();result['method']='cal.'+policy;result['calibration_cutoff']='';result['adjusted']=False
        prior={}
        for j,cutoff in enumerate(ends):
            end=ends[j+1] if j+1<len(ends) else cfg()['data_end']
            params,rows=decision(bank,cutoff,policy)
            for row in rows:
                state=row['state'];dates=set(row['validation_dates']);old_dates=prior.get(state,set())
                reuse.append(dict(policy=policy,cutoff=cutoff,state=state,current_n=len(dates),
                    reused_n=len(dates&old_dates),new_n=len(dates-old_dates)))
                prior[state]=dates
                use=result.date.gt(cutoff)&result.date.le(end)
                if state!=-1:use &=result.state.eq(state)
                row['next_period_covered_dates']=result.loc[use,'date'].tolist()
                row['actual_adjusted_dates']=result.loc[use,'date'].tolist() if state in params else []
                if state in params:
                    slope,delta=params[state];base=bank.loc[use,'probability'].to_numpy()
                    result.loc[use,'probability']=expit(slope*logit(np.clip(base,1e-12,1-1e-12))+delta)
                    result.loc[use,'adjusted']=True
                result.loc[use,'calibration_cutoff']=cutoff
                decisions.append(row)
        outputs.append(result)
    return pd.concat(outputs,ignore_index=True),decisions,pd.DataFrame(reuse)
