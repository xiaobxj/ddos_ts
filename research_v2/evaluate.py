"""Paired exploratory development statistics. Identical execution intervals for every arm."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from core import ROOT,V1
import importlib.util

OUT=ROOT/'results'
spec=importlib.util.spec_from_file_location('v1_evaluation',V1/'evaluate.py')
v1_eval=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1_eval)


def metrics(g):
    y=g.actual.to_numpy(float)
    yhat=g.predicted_return.to_numpy(float)
    pos=g.position.to_numpy(int)
    actual_up=y>0
    correct=pos==actual_up
    p=g.probability.to_numpy(float)
    pc=np.clip(p,1e-12,1-1e-12)
    probability_valid=np.isfinite(p).all()
    return dict(n=len(g),accuracy=float(correct.mean()),
                balanced_accuracy=float((pos[actual_up].mean()+(1-pos[~actual_up]).mean())/2),
                true_up_rate=float(actual_up.mean()),predicted_up_rate=float(pos.mean()),
                return_rmse=float(np.sqrt(np.mean((yhat-y)**2))),
                mse_skill_vs_zero=float(1-np.mean((yhat-y)**2)/np.mean(y*y)),
                return_correlation=float(np.corrcoef(yhat,y)[0,1]) if yhat.std()>1e-12 else None,
                conditional_return_long=float(y[pos==1].mean()) if (pos==1).any() else None,
                conditional_return_cash=float(y[pos==0].mean()) if (pos==0).any() else None,
                logloss=float(-np.mean(actual_up*np.log(pc)+(1-actual_up)*np.log1p(-pc))) if probability_valid else None,
                brier=float(np.mean((p-actual_up)**2)) if probability_valid else None)


def weekly_returns(g,bps):
    g=g.sort_values('anchor')
    assert np.array_equal(g.exit.to_numpy()[:-1],g.entry.to_numpy()[1:])
    pos=g.position.to_numpy(int)
    turn=np.abs(np.diff(np.r_[0,pos]))
    growth=(1-turn*bps/10000)*(1+pos*g.actual.to_numpy(float))
    growth[-1]*=(1-pos[-1]*bps/10000)
    return growth-1


def bootstrap_indices(n,cfg):
    block=cfg['block_observations']; b=cfg['replicates']
    rng=np.random.default_rng(cfg['seed'])
    starts=rng.integers(n,size=(b,int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(b,-1)[:,:n]


def mean_diff_stats(d,ids):
    observed=d.mean()
    boot=d[ids].mean(axis=1)
    centered=(d-observed)[ids].mean(axis=1)
    p=(1+np.sum(abs(centered)>=abs(observed)))/(len(ids)+1)
    return dict(difference=float(observed),ci95_low=float(np.quantile(boot,.025)),
                ci95_high=float(np.quantile(boot,.975)),p=float(p))


def paired(g,ref,cfg):
    g=g.sort_values('date');ref=ref.sort_values('date')
    assert g.date.tolist()==ref.date.tolist()
    ids=bootstrap_indices(len(g),cfg)
    d=g.correct.to_numpy(float)-ref.correct.to_numpy(float)
    err=(g.predicted_return.to_numpy()-g.actual.to_numpy())**2
    other=(ref.predicted_return.to_numpy()-ref.actual.to_numpy())**2
    rmses=np.sqrt(err[ids].mean(axis=1))-np.sqrt(other[ids].mean(axis=1))
    return dict(method=g.method.iloc[0],reference=ref.method.iloc[0],n=len(g),
                accuracy=mean_diff_stats(d,ids),mse=mean_diff_stats(err-other,ids),
                rmse=dict(difference=float(np.sqrt(err.mean())-np.sqrt(other.mean())),
                          ci95_low=float(np.quantile(rmses,.025)),ci95_high=float(np.quantile(rmses,.975))),
                mean_weekly_net_return_10bps=mean_diff_stats(weekly_returns(g,10)-weekly_returns(ref,10),ids))


def holm(pvalues):
    p=np.asarray(pvalues)
    order=np.argsort(p)
    result=np.empty(len(p))
    last=0.
    for i,j in enumerate(order):
        last=max(last,min(1.,float(p[j])*(len(p)-i)))
        result[j]=last
    return result.tolist()


def main():
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'selection.json').read_text())
    frame=pd.read_csv(V1/'data/1_000300.csv')
    pred=pd.read_csv(OUT/'predictions.csv')
    groups={m:g.sort_values('date') for m,g in pred.groupby('method',sort=False)}
    summary=[];yearly=[];costs=[];curves=[];weekly=[]
    for name,g in groups.items():
        assert len(g)==272 and g.date.is_unique
        assert g.exit.iloc[-1]==len(frame)-1
        summary.append(dict(method=name,**metrics(g)))
        for year,h in g.groupby('year'):
            yearly.append(dict(method=name,year=int(year),**metrics(h)))
        for bps in cfg['cost_bps']:
            m,rows=v1_eval.backtest(frame,g,bps)
            costs.append(dict(method=name,**m))
            w=weekly_returns(g,bps)
            assert abs(np.prod(1+w)-1-m['total_return'])<1e-10
            if bps==10:
                curves.extend([dict(method=name,**r) for r in rows])
                weekly.extend([dict(method=name,date=g.date.iloc[i],net_return=float(r)) for i,r in enumerate(w)])
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'cost_sensitivity.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'equity_curves_10bps.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_net_returns_10bps.csv',index=False)
    comparisons=[paired(groups[a],groups[b],cfg['bootstrap']) for a,b in cfg['prespecified_comparisons']]
    for result,adjusted in zip(comparisons,holm([r['mse']['p'] for r in comparisons])):
        result['mse']['holm_adjusted_p']=adjusted
    (OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    chosen=selection['validation_selected_pipeline']
    chosen_refs=[paired(groups[chosen],groups[r],cfg['bootstrap']) for r in ['b_exec_annual','v1_adaptive_normalized','always_up']]
    (OUT/'selected_pipeline_comparisons.json').write_text(json.dumps(chosen_refs,indent=2),encoding='utf-8')
    validation=pd.read_csv(OUT/'validation_predictions.csv')
    valrows=[]
    for method,data in selection['selected_lambdas'].items():
        g=validation[(validation.method==method)&np.isclose(validation.lam,data['lam'])]
        valrows.append(dict(method=method,lam=data['lam'],n=len(g),rmse=data['return_rmse'],
                            accuracy=float((g.position==(g.actual>0)).mean()),criterion=data['criterion']))
    pd.DataFrame(valrows).to_csv(OUT/'validation_summary.csv',index=False)
    # Numeric diagnostics on frozen predictions; never select another pipeline here.
    common=groups['b_exec_annual']
    old=groups['v1_adaptive_normalized']
    audit=dict(classification=cfg['classification'],n=272,methods=len(groups),
                signal_start=common.date.iloc[0],signal_end=common.date.iloc[-1],
                entry_start=frame.date.iloc[common.entry.iloc[0]],exit_end=common.exit_date.iloc[-1],
                close_vs_execution_sign_disagreements=int(((common.actual>0)!=(common.close_actual>0)).sum()),
                v1_accuracy_on_execution_labels=float(old.correct.mean()),
                validation_selected_pipeline=chosen,
                weekly_daily_accounting_reconciled=True,
                zero_return_rmse=float(np.sqrt(np.mean(common.actual.to_numpy()**2))))
    (OUT/'evaluation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    table=summary.merge(pd.DataFrame(costs).query('cost_bps==10'),on='method')
    print(table[['method','accuracy','return_rmse','return_correlation','total_return','max_drawdown','exposure']].to_string(index=False))
    print('Validation-selected pipeline:',chosen)
    for x in comparisons:
        print(x['method'],'vs',x['reference'],'rmse difference',x['rmse'],'adjusted p',x['mse']['holm_adjusted_p'])
    for x in chosen_refs:
        print('SELECTED',x['reference'],x['accuracy'],x['mean_weekly_net_return_10bps'])


if __name__=='__main__':
    main()
