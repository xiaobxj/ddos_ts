"""Identical-window predictive metrics, next-open costs and paired uncertainty."""
from common import *
import importlib.util

spec = importlib.util.spec_from_file_location('v1_evaluate', V1 / 'evaluate.py')
v1_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1_eval)


def metrics(g):
    y = g.actual.to_numpy(float); pred = g.predicted_return.to_numpy(float)
    pos = g.position.to_numpy(int); up = y > 0
    return dict(n=len(g), accuracy=float((pos == up).mean()),
                balanced_accuracy=float((pos[up].mean() + (1-pos[~up]).mean()) / 2) if up.any() and (~up).any() else None,
                actual_up_rate=float(up.mean()), predicted_up_rate=float(pos.mean()),
                return_rmse=float(np.sqrt(np.mean((pred-y)**2))),
                mse_skill_vs_zero=float(1-np.mean((pred-y)**2)/np.mean(y*y)),
                correlation=float(np.corrcoef(pred,y)[0,1]) if pred.std()>1e-12 else None,
                mean_forecast=float(pred.mean()), forecast_std=float(pred.std()),
                mean_actual=float(y.mean()),
                mean_return_held=float(y[pos==1].mean()) if (pos==1).any() else None,
                mean_return_unheld=float(y[pos==0].mean()) if (pos==0).any() else None)


def weekly_returns(g, bps):
    g = g.sort_values('anchor')
    assert np.array_equal(g.exit.to_numpy()[:-1], g.entry.to_numpy()[1:])
    pos = g.position.to_numpy(int)
    turnover = np.abs(np.diff(np.r_[0, pos]))
    growth = (1-turnover*bps/10000)*(1+pos*g.actual.to_numpy(float))
    growth[-1] *= 1-pos[-1]*bps/10000
    return growth-1


def bootstrap_indices(n):
    config = cfg()['bootstrap']; block = config['block_observations']
    rng = np.random.default_rng(config['seed'])
    starts = rng.integers(n, size=(config['replicates'], int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(config['replicates'],-1)[:,:n]


def difference(values, ids):
    mean = values.mean(); boot = values[ids].mean(axis=1)
    centered = (values-mean)[ids].mean(axis=1)
    p = (1+np.sum(abs(centered)>=abs(mean)))/(len(ids)+1)
    return dict(difference=float(mean), ci95_low=float(np.quantile(boot,.025)),
                ci95_high=float(np.quantile(boot,.975)), p=float(p))


def paired(g, ref):
    g=g.sort_values('date'); ref=ref.sort_values('date')
    assert g.date.tolist()==ref.date.tolist()
    ids=bootstrap_indices(len(g))
    err=(g.predicted_return.to_numpy()-g.actual.to_numpy())**2
    err_ref=(ref.predicted_return.to_numpy()-ref.actual.to_numpy())**2
    rmse=np.sqrt(err[ids].mean(axis=1))-np.sqrt(err_ref[ids].mean(axis=1))
    return dict(method=g.method.iloc[0], reference=ref.method.iloc[0], n=len(g),
                accuracy=difference(g.correct.to_numpy(float)-ref.correct.to_numpy(float),ids),
                mse=difference(err-err_ref,ids),
                rmse=dict(difference=float(np.sqrt(err.mean())-np.sqrt(err_ref.mean())),
                          ci95_low=float(np.quantile(rmse,.025)),ci95_high=float(np.quantile(rmse,.975))),
                mean_weekly_net_10bps=difference(weekly_returns(g,10)-weekly_returns(ref,10),ids))


def holm(ps):
    order=np.argsort(ps); result=np.empty(len(ps)); last=0.
    for i,j in enumerate(order):
        last=max(last,min(1.,ps[j]*(len(ps)-i)));result[j]=last
    return result.tolist()


def influence(g, ref):
    net=weekly_returns(g,10); other=weekly_returns(ref,10)
    contribution=np.log1p(net)-np.log1p(other)
    order=np.argsort(-contribution)
    total=float(contribution.sum())
    return dict(method=g.method.iloc[0], reference=ref.method.iloc[0],
                total_log_wealth_gap=total, largest_positive_date=g.date.iloc[order[0]],
                largest_positive_log_contribution=float(contribution[order[0]]),
                gap_excluding_largest_positive=total-float(contribution[order[0]]),
                gap_excluding_top_three=total-float(contribution[order[:3]].sum()),
                different_position_weeks=int((g.position.to_numpy()!=ref.position.to_numpy()).sum()))


def main():
    local = pd.read_csv(OUT/'local_predictions.csv')
    kronos = pd.read_csv(OUT/'kronos_predictions.csv')
    manifest = json.loads((OUT/'kronos_run_manifest.json').read_text(encoding='utf-8'))
    assert manifest.get('finished_utc'), 'Kronos inference must finish before scoring'
    pred = pd.concat([local,kronos],ignore_index=True)
    pred.to_csv(OUT/'predictions.csv',index=False)
    frame = pd.read_csv(V1/'data/1_000300.csv')
    all_metrics=[]; costs=[]; yearly=[]; curves=[]; weekly=[]; pairs=[]; impacts=[]; diagnostics=[]
    for window,start in cfg()['evaluation_windows'].items():
        groups={name:g.sort_values('anchor') for name,g in pred[pred.date>=start].groupby('method',sort=False)
                if window!='development_full' or not name.startswith('kronos')}
        dates=groups['ridge_target'].date.tolist()
        for name,g in groups.items():
            assert g.date.tolist()==dates
            all_metrics.append(dict(window=window,method=name,**metrics(g)))
            net=weekly_returns(g,10)
            for year,h in g.groupby('year'):
                mask=(g.year==year).to_numpy()
                yearly.append(dict(window=window,method=name,year=int(year),**metrics(h),
                                   period_net_return_10bps=float(np.prod(1+net[mask])-1)))
            for bps in cfg()['cost_bps']:
                m,rows=v1_eval.backtest(frame,g,bps)
                w=weekly_returns(g,bps)
                assert abs(np.prod(1+w)-1-m['total_return'])<1e-10
                costs.append(dict(window=window,method=name,**m))
                if bps==10:
                    curves.extend([dict(window=window,method=name,**row) for row in rows])
                    weekly.extend([dict(window=window,method=name,date=date,net_return=float(value)) for date,value in zip(g.date,w)])
        comparisons=[(a,b) for a,b in cfg()['primary_comparisons'] if a in groups and b in groups]
        current=[dict(window=window,**paired(groups[a],groups[b])) for a,b in comparisons]
        for item,p in zip(current,holm([p['mse']['p'] for p in current])):
            item['mse']['holm_adjusted_p']=p
        pairs.extend(current)
        impact_pairs=comparisons+[(name,'buy_hold') for name in ['hgb_target','ridge_cross','hgb_cross','kronos_ohlcv'] if name in groups]
        impacts.extend([dict(window=window,**influence(groups[a],groups[b])) for a,b in impact_pairs])
        diagnostic_pairs=[(a,'zero_return_cash') for a in groups if a not in ['zero_return_cash','buy_hold','round2_frozen','training_mean']]
        diagnostic_pairs += [('ridge_target_matched','ridge_target'),('hgb_target_matched','hgb_target')]
        if 'kronos_ohlcv' in groups:
            diagnostic_pairs.append(('kronos_ohlc_diagnostic','kronos_ohlcv'))
        diagnostics.extend([dict(window=window,**paired(groups[a],groups[b])) for a,b in diagnostic_pairs])
    pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'cost_sensitivity.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'equity_curves_10bps.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_net_returns_10bps.csv',index=False)
    save_json(OUT/'paired_comparisons.json',pairs)
    save_json(OUT/'diagnostic_comparisons.json',diagnostics)
    save_json(OUT/'influence_diagnostics.json',impacts)
    validation=pd.read_csv(OUT/'validation_selected_predictions.csv')
    pd.DataFrame([dict(method=name,**metrics(g)) for name,g in validation.groupby('method')]).to_csv(OUT/'validation_summary.csv',index=False)
    seeds=pd.read_csv(OUT/'kronos_seed_predictions.csv')
    seedmetrics=[]; seeddifferences=[]
    for window,start in list(cfg()['evaluation_windows'].items())[1:]:
        for (name,seed),g in seeds[seeds.date>=start].groupby(['method','base_seed']):
            m,_=v1_eval.backtest(frame,g.sort_values('date'),10)
            seedmetrics.append(dict(window=window,method=name,seed=int(seed),**metrics(g),**m))
        for name,g in seeds[seeds.date>=start].groupby('method'):
            pivot=g.pivot(index='date',columns='base_seed',values='predicted_return')
            agreement=(pivot>0).nunique(axis=1)==1
            seeddifferences.append(dict(window=window,method=name,anchors=len(pivot),
                                       unanimous_direction_rate=float(agreement.mean()),
                                       mean_across_seed_forecast_std=float(pivot.std(axis=1).mean())))
    pd.DataFrame(seedmetrics).to_csv(OUT/'kronos_seed_metrics.csv',index=False)
    save_json(OUT/'kronos_sampling_diagnostics.json',seeddifferences)
    quality=[dict(method=name,bars=int(g.horizon.sum()),invalid_bars=int(g.invalid_bars.sum()),
                  invalid_fraction=float(g.invalid_bars.sum()/g.horizon.sum())) for name,g in kronos.groupby('method')]
    save_json(OUT/'kronos_output_quality.json',quality)
    table=pd.DataFrame(all_metrics).merge(pd.DataFrame(costs).query('cost_bps==10'),on=['window','method'])
    print(table[['window','method','n','accuracy','return_rmse','correlation','total_return','max_drawdown','exposure']].to_string(index=False))


if __name__=='__main__':
    main()
