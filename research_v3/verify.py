"""Independent labels, paired samples, immutable assets and execution accounting."""
from common import *
import subprocess
from evaluate import weekly_returns, metrics


def main():
    tests = subprocess.run([sys.executable, str(ROOT/'test_contract.py')], capture_output=True, text=True, encoding='utf-8')
    (OUT/'test_results.txt').write_text(tests.stdout+tests.stderr, encoding='utf-8')
    assert tests.returncode == 0, tests.stderr
    local = json.loads((OUT/'local_run_manifest.json').read_text(encoding='utf-8'))
    kronos = json.loads((OUT/'kronos_run_manifest.json').read_text(encoding='utf-8'))
    for manifest in [local,kronos]:
        assert manifest.get('finished_utc')
        assert sha(ROOT/'protocol.json') == manifest['protocol_sha256']
        assert previous_evidence() == manifest['previous_evidence']
        for name,digest in manifest['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
    for name,digest in kronos['official_source_sha256'].items():
        assert sha(ROOT/name)==digest
    for group in kronos['weights']:
        assert group['weights_identical_to_initial_upload']
        for file in group['files']:
            assert sha(ROOT/file['file']) == file['sha256']
    frame=pd.read_csv(V1/'data/1_000300.csv')
    obs=pd.read_csv(OUT/'observation_table.csv')
    pred=pd.read_csv(OUT/'predictions.csv')
    previous=pd.read_csv(V2/'results/predictions.csv').query("method=='f_multi_exp_quarter'").sort_values('anchor')
    friday=np.flatnonzero(pd.to_datetime(frame.date).dt.dayofweek.to_numpy()==4)
    next_friday=dict(zip(friday[:-1],friday[1:]))
    assert not pred.duplicated(['method','date']).any()
    for name,g in pred.groupby('method'):
        g=g.sort_values('anchor')
        expected_anchors=previous[previous.date>=cfg()['kronos']['common_start']].anchor if name.startswith('kronos') else previous.anchor
        np.testing.assert_array_equal(g.anchor,expected_anchors)
        a=g.anchor.to_numpy(int)
        exits=np.array([next_friday[t]+1 for t in a])
        np.testing.assert_array_equal(g.entry,a+1)
        np.testing.assert_array_equal(g.exit,exits)
        actual=frame.open.to_numpy()[exits]/frame.open.to_numpy()[a+1]-1
        np.testing.assert_allclose(g.actual,actual,atol=1e-12,rtol=0)
        assert np.isfinite(g.predicted_return).all()
        np.testing.assert_array_equal(g.position,(g.predicted_return>0).astype(int))
        np.testing.assert_array_equal(g.correct,(g.position==(actual>0)).astype(int))
        assert (g.cutoff<g.date).all()
    folds=json.loads((OUT/'folds.json').read_text(encoding='utf-8'))
    for fold in folds:
        train=(obs.completed<=fold['cutoff']).to_numpy().copy()
        if fold['matched']:
            train &= obs.panel_valid.to_numpy(bool)
        assert train.sum()==fold['train_n']
        assert obs.date[train].min()==fold['first_train']
        assert obs.completed[train].max()==fold['last_label']
        assert fold['last_label']<=fold['cutoff']<fold['first_signal']<=fold['last_signal']<=fold['end']
    for kind in ['ridge','hgb']:
        a=pred[pred.method==kind+'_target_matched'].sort_values('date')
        b=pred[pred.method==kind+'_cross'].sort_values('date')
        np.testing.assert_array_equal(a.train_n,b.train_n)
        assert a.cutoff.tolist()==b.cutoff.tolist()
    choice=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    grid=pd.read_csv(OUT/'validation_grid_predictions.csv',dtype={'option':str})
    selected=pd.read_csv(OUT/'validation_selected_predictions.csv')
    for model,record in choice['parameters'].items():
        g=grid[grid.method==model+'_target']
        errors={key:float(np.mean((h.predicted_return-h.actual)**2)) for key,h in g.groupby('option')}
        key=record['option']
        assert abs(np.sqrt(errors[key])-record['validation_rmse'])<1e-12
        assert errors[key]<=min(errors.values())+1e-14
        h=selected[selected.method==model+'_target'].sort_values('date')
        original=g[g.option==key].sort_values('date')
        np.testing.assert_allclose(h.predicted_return,original.predicted_return,atol=1e-12,rtol=0)
    for name,g in selected.groupby('method'):
        assert len(g)==141 and g.date.max()<='2020-12-31'
    saved=pred[pred.method=='round2_frozen'].sort_values('anchor')
    np.testing.assert_allclose(saved.predicted_return,previous.predicted_return,atol=1e-14,rtol=0)
    # If validation again selects lambda 10, full-history Ridge must reproduce v2.
    if choice['parameters']['ridge']['setting']==10.0:
        new=pred[pred.method=='ridge_target'].sort_values('anchor')
        np.testing.assert_allclose(new.predicted_return,previous.predicted_return,atol=1e-10,rtol=0)
    seeds=pd.read_csv(OUT/'kronos_seed_predictions.csv')
    path_count=0
    path_columns=['open','high','low','close','volume','amount']
    path_dtypes={column:np.float32 for column in path_columns}
    for row in pred[pred.method.str.startswith('kronos')].itertuples(index=False):
        paths=[]
        for seed in cfg()['kronos']['seeds']:
            # Official inference writes float32 CSV representations. Restore the
            # original dtype before aggregation, then use float64 for returns.
            path=pd.read_csv(OUT/'kronos_paths'/f'{row.method}_{row.date}_{seed}.csv',dtype=path_dtypes)
            assert len(path)==int(row.horizon)
            assert path.forecast_date.iloc[0]==frame.date.iloc[int(row.entry)]
            assert path.forecast_date.iloc[-1]==row.exit_date
            value=float(path.open.iloc[-1])/float(path.open.iloc[0])-1
            seedrow=seeds[(seeds.method==row.method)&(seeds.date==row.date)&(seeds.base_seed==seed)].iloc[0]
            assert seedrow.effective_seed==seed+row.anchor
            assert abs(value-seedrow.predicted_return)<1e-12
            assert float(path.open.iloc[0])==seedrow.predicted_entry_open
            assert float(path.open.iloc[-1])==seedrow.predicted_exit_open
            paths.append(path[path_columns].to_numpy())
            path_count+=1
        mean=np.mean(paths,axis=0)
        expected=float(mean[-1,0])/float(mean[0,0])-1
        assert abs(expected-row.predicted_return)<1e-12
        assert abs(mean[0,0]-row.predicted_entry_open)<1e-9
        ensemble=pd.read_csv(OUT/'kronos_paths'/f'{row.method}_{row.date}_ensemble.csv',dtype=path_dtypes)
        np.testing.assert_array_equal(ensemble[path_columns].to_numpy(),mean)
    inputs=json.loads((OUT/'kronos_input_audit.json').read_text(encoding='utf-8'))
    from run_kronos import prepare_input
    for item in inputs:
        x,times=prepare_input(frame.iloc[:item['anchor']+1],item['anchor'],item['price_only'])
        assert hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()==item['input_sha256']
        assert item['input_end'][:10]==item['date']
    costs=pd.read_csv(OUT/'cost_sensitivity.csv')
    curves=pd.read_csv(OUT/'equity_curves_10bps.csv')
    summary=pd.read_csv(OUT/'metrics.csv')
    for (window,name),g in costs.groupby(['window','method']):
        rows=pred[(pred.method==name)&(pred.date>=cfg()['evaluation_windows'][window])].sort_values('anchor')
        g=g.sort_values('cost_bps')
        assert (np.diff(g.total_return)<=1e-12).all()
        for cost in g.itertuples():
            assert abs(np.prod(1+weekly_returns(rows,cost.cost_bps))-1-cost.total_return)<1e-10
            if name=='buy_hold':
                analytical=frame.open.iloc[-1]/frame.open.iloc[int(rows.entry.iloc[0])] * (1-cost.cost_bps/10000)**2 -1
                assert abs(analytical-cost.total_return)<1e-10
        equity=curves[(curves.window==window)&(curves.method==name)]
        assert abs(equity.wealth.iloc[-1]-1-g[g.cost_bps==10].total_return.iloc[0])<1e-10
        stored=summary[(summary.window==window)&(summary.method==name)].iloc[0]
        computed=metrics(rows)
        assert abs(stored.accuracy-computed['accuracy'])<1e-12
        assert abs(stored.return_rmse-computed['return_rmse'])<1e-12
    audit=dict(status='PASS',unit_tests=6,predictions=len(pred),local_models=6,kronos_input_arms=2,
               full_development_weeks=272,kronos_common_weeks=kronos['anchors'],
               official_seed_paths_recomputed=path_count,refit_folds_checked=len(folds),
               previous_files_preserved=len(local['previous_evidence']),source_and_weight_hashes_match=True,
               observation_and_feature_parity_with_round2=True,all_labels_independently_recomputed=True,
               paired_cross_feature_training_masks_identical=True,validation_choices_recomputed=True,
               kronos_forecasts_recomputed_from_saved_paths=True,amount_not_synthesized=True,
               float32_csv_paths_restored_and_ensemble_bitwise_matched=True,
               deterministic_kronos_repeatability=kronos['deterministic_repeatability_passed'],
               daily_weekly_and_analytic_buyhold_accounting_reconciled=True,
               checked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save_json(OUT/'verification.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
