"""Paired evaluation using the unchanged round-3 accounting/statistic functions."""
from util import *
import importlib.util

spec=importlib.util.spec_from_file_location('round3_evaluation',V3/'evaluate.py')
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)
assert v3.cfg()['bootstrap']==dict(block_observations=8,replicates=10000,seed=20260910)


def main():
    run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    assert run.get('finished_utc')
    config=cfg();pred=pd.read_csv(OUT/'predictions.csv')
    frame=pd.read_csv(V1/'data/1_000300.csv')
    metrics=[];costs=[];years=[];curves=[];weekly=[];pairs=[];diagnostics=[];impacts=[]
    for window,start in config['evaluation_windows'].items():
        groups={name:g.sort_values('date') for name,g in pred[pred.date>=start].groupby('method',sort=False)}
        dates=groups['raw_fixed'].date.tolist()
        for name,g in groups.items():
            assert g.date.tolist()==dates
            metrics.append(dict(window=window,method=name,**v3.metrics(g)))
            net=v3.weekly_returns(g,10)
            for year,h in g.groupby('year'):
                years.append(dict(window=window,method=name,year=int(year),**v3.metrics(h),
                                  period_net_return_10bps=float(np.prod(1+net[(g.year==year).to_numpy()])-1)))
            for bps in config['cost_bps']:
                result,equity=v3.v1_eval.backtest(frame,g,bps)
                assert abs(np.prod(1+v3.weekly_returns(g,bps))-1-result['total_return'])<1e-10
                costs.append(dict(window=window,method=name,**result))
                if bps==10:
                    curves.extend([dict(window=window,method=name,**r) for r in equity])
                    weekly.extend([dict(window=window,method=name,date=d,net_return=float(r)) for d,r in zip(g.date,net)])
        family=[dict(window=window,**v3.paired(groups[a],groups[b])) for a,b in config['primary_comparisons']]
        for item,p in zip(family,v3.holm([x['mse']['p'] for x in family])):
            item['mse']['holm_adjusted_p']=p
        pairs.extend(family)
        comparisons=config['primary_comparisons']+[(a['name'],'ridge_matched_schedule') for a in config['arms']]
        for a,b in comparisons:
            impacts.append(dict(window=window,**v3.influence(groups[a],groups[b])))
        for a in [arm['name'] for arm in config['arms']]:
            for b in ['ridge_matched_schedule','zero_return_cash','buy_hold']:
                diagnostics.append(dict(window=window,**v3.paired(groups[a],groups[b])))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'cost_sensitivity.csv',index=False)
    pd.DataFrame(years).to_csv(OUT/'yearly_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'equity_curves_10bps.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_net_returns_10bps.csv',index=False)
    save_json(OUT/'paired_comparisons.json',pairs)
    save_json(OUT/'diagnostic_comparisons.json',diagnostics)
    save_json(OUT/'influence_diagnostics.json',impacts)
    seed=pd.read_csv(OUT/'development_seed_predictions.csv')
    seed_metrics=[];stability=[]
    for window,start in config['evaluation_windows'].items():
        for (name,number),g in seed[seed.date>=start].groupby(['method','seed']):
            result,_=v3.v1_eval.backtest(frame,g.sort_values('date'),10)
            seed_metrics.append(dict(window=window,method=name,seed=int(number),**v3.metrics(g),**result))
        for name,g in seed[seed.date>=start].groupby('method'):
            matrix=g.pivot(index='date',columns='seed',values='predicted_return')
            stability.append(dict(window=window,method=name,unanimous_direction_rate=float(((matrix>0).nunique(axis=1)==1).mean()),
                                  mean_across_seed_forecast_std=float(matrix.std(axis=1).mean())))
    pd.DataFrame(seed_metrics).to_csv(OUT/'seed_metrics.csv',index=False)
    save_json(OUT/'seed_stability.json',stability)
    table=pd.DataFrame(metrics).merge(pd.DataFrame(costs).query('cost_bps==10'),on=['window','method'])
    print(table[['window','method','n','accuracy','return_rmse','correlation','total_return','max_drawdown','exposure']].to_string(index=False))


if __name__=='__main__':
    main()
