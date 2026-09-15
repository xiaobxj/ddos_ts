"""Fixed cohort accounting; one delayed primary comparison family."""
from common49 import *

def comparison(losses,settings):
    a=np.asarray(losses,float);valid=np.isfinite(a);guard(valid.sum()>0,'No paired observations')
    n=len(a);mu=float(np.nanmean(a));rng=np.random.default_rng(settings['seed'])
    starts=rng.integers(n,size=(settings['bootstrap_replicates'],int(np.ceil(n/settings['block_weeks']))))
    ids=((starts[:,:,None]+np.arange(settings['block_weeks']))%n).reshape(len(starts),-1)[:,:n]
    samples=a[ids];counts=np.isfinite(samples).sum(1);guard((counts>0).all(),'Empty bootstrap block sample')
    means=np.nansum(samples,axis=1)/counts;center=means-mu
    return dict(n=int(valid.sum()),difference=mu,ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),p=float((1+int((abs(center)>=abs(mu)).sum()))/(len(means)+1)))
def summarize(events,at):
    protocol=cfg();pred={e['key']:e for e in events if e['kind']=='prediction'};labels={e['key']:e for e in events if e['kind']=='label'}
    coverage=[];complete=[];metrics=[]
    for date in protocol['signal_slots']:
        if date in labels:
            status='COMPLETE' if labels[date]['payload']['status']=='MATURE' else 'INVALID_LABEL'
        elif date in pred:status='PENDING_LABEL'
        elif at.astimezone(TZ)<datetime.fromisoformat(date+'T23:00:00+08:00'):status='NOT_DUE'
        else:status='NO_TIMELY_PREDICTION'
        coverage.append(dict(date=date,status=status))
        if status=='COMPLETE':complete.append(date)
    for h in HISTORIES:
        for method in METHODS:
            ps=[];ys=[]
            for date in complete:
                row=next(r for r in pred[date]['payload']['forecast']['ensemble_predictions'] if r['history']==h and r['method']==method)
                ps.append(row['probability']);ys.append(labels[date]['payload']['actual_up'])
            if ps:
                a=np.array(ps);y=np.array(ys);metrics.append(dict(history=h,method=method,n=len(a),correct=int(((a>.5)==y).sum()),accuracy=float(((a>.5)==y).mean()),brier=float(((a-y)**2).mean())))
    earliest=protocol['cohort']['primary_review_not_before'];pending=sum(r['status']=='PENDING_LABEL' for r in coverage)
    allowed=at.astimezone(TZ).strftime('%Y-%m-%d')>=earliest and len(complete)>=protocol['cohort']['minimum_complete_mature_signals'] and pending==0
    pairs=[]
    if allowed:
        for c in protocol['comparisons']:
            losses=[]
            for date in protocol['signal_slots']:
                if date not in complete:losses.append(np.nan);continue
                rows=pred[date]['payload']['forecast']['ensemble_predictions'];y=labels[date]['payload']['actual_up']
                p=next(r['probability'] for r in rows if r['history']==c['history'] and r['method']==c['method'])
                q=next(r['probability'] for r in rows if r['history']==c['reference_history'] and r['method']==c['method'])
                losses.append(float(((p>.5)!=y)-int((q>.5)!=y)) if c['metric']=='direction_error' else (p-y)**2-(q-y)**2)
            pairs.append(dict(c,**comparison(losses,protocol['inference'])))
        last=0.
        for rank,i in enumerate(sorted(range(len(pairs)),key=lambda i:pairs[i]['p'])):
            last=max(last,min(1.,pairs[i]['p']*(len(pairs)-rank)));pairs[i]['holm_adjusted_p']=last
    return dict(generated_utc=iso(at),status='PRIMARY_READY' if allowed else 'DESCRIPTIVE_ONLY',primary_review_not_before=earliest,complete_mature_signals=len(complete),recorded_predictions=len(pred),coverage=coverage,metrics=metrics,comparisons=pairs,limitations='Fixed recipes and same future cohort; local timestamps only. No trading PnL. Missing slots are explicit; observed subset can still be selective.')
