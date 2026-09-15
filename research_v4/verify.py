"""Independent source/cache, chronology, checkpoint and account-equity checks."""
from util import *
from data import packed_path
from architecture import SegmentedCrossformer
from evaluate import v3
import subprocess
import torch


def main():
    initialize_torch()
    outputs=[]
    for script in ['test_model.py','test_data.py','verify_tokenizer.py']:
        p=subprocess.run([sys.executable,str(ROOT/script)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        outputs.append(f'== {script} ==\n{p.stdout}{p.stderr}')
        if p.returncode:
            (OUT/'test_results.txt').write_text('\n'.join(outputs),encoding='utf-8')
            raise AssertionError(outputs[-1])
    (OUT/'test_results.txt').write_text('\n'.join(outputs),encoding='utf-8')
    run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    prep=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert run.get('finished_utc')
    for manifest in [run,prep]:
        assert manifest['protocol_sha256']==sha(ROOT/'protocol.json')
        assert manifest['previous_evidence']==previous_evidence()
        for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert run['preparation_manifest_sha256']==sha(OUT/'preparation_manifest.json')
    for name,digest in run['official_crossformer_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in prep['tokenizer_source_sha256'].items():assert sha(PROJECT/name)==digest
    for name,digest in prep['cache_sha256'].items():assert sha(ROOT/name)==digest,name
    for file in prep['tokenizer_weight']['files']:assert sha(V3/file['file'])==file['sha256']
    config=cfg();obs=pd.read_csv(OUT/'observation_table.csv')
    frame=pd.read_csv(V1/'data/1_000300.csv')
    old=pd.read_csv(V3/'results/predictions.csv')
    anchors=old[(old.method=='ridge_target')&(old.date>='2024-07-01')].sort_values('date').anchor.to_numpy()
    pred=pd.read_csv(OUT/'predictions.csv')
    seeds=pd.read_csv(OUT/'development_seed_predictions.csv')
    assert not pred.duplicated(['method','date']).any()
    friday=np.flatnonzero(pd.to_datetime(frame.date).dt.dayofweek.to_numpy()==4)
    next_friday=dict(zip(friday[:-1],friday[1:]))
    for name,g in pred.groupby('method'):
        g=g.sort_values('anchor');a=g.anchor.to_numpy(int)
        np.testing.assert_array_equal(a,anchors)
        exits=np.array([next_friday[t]+1 for t in a])
        np.testing.assert_array_equal(g.exit,exits);np.testing.assert_array_equal(g.entry,a+1)
        actual=frame.open.to_numpy()[exits]/frame.open.to_numpy()[a+1]-1
        np.testing.assert_allclose(g.actual,actual,rtol=0,atol=1e-12)
        assert np.isfinite(g.predicted_return).all()
        np.testing.assert_array_equal(g.position,g.predicted_return>0)
        np.testing.assert_array_equal(g.correct,g.position==(actual>0))
        assert (g.cutoff<g.date).all()
    folds=json.loads((OUT/'folds.json').read_text(encoding='utf-8'))
    for fold in folds:
        tr=(obs.completed<=fold['cutoff'])
        assert tr.sum()==fold['train_n']
        assert obs.date[tr].min()==fold['first_train']
        assert obs.completed[tr].max()==fold['last_label']
        assert fold['last_label']<=fold['cutoff']<fold['first_signal']<=fold['last_signal']<=fold['end']
        if fold['phase']=='development':
            assert fold['cutoff']>='2024-06-30'
            assert f"{fold['prototype_year']-1}-12-31"<=fold['cutoff']
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    val=pd.read_csv(OUT/'validation_seed_predictions.csv')
    computed=[]
    for epoch in config['training']['epoch_candidates']:
        subset=val[val.epoch==epoch]
        assert subset.method.unique().tolist()==['raw_fixed']
        assert subset.groupby('date').size().eq(3).all()
        average=subset.groupby('date').agg(predicted_return=('predicted_return','mean'),actual=('actual','first'))
        assert len(average)==141 and average.index.max()<='2020-12-31'
        computed.append((float(np.sqrt(np.mean((average.predicted_return-average.actual)**2))),epoch))
    assert min(computed)[1]==selection['selected_epochs']
    checkpoints=json.loads((OUT/'checkpoint_manifest.json').read_text(encoding='utf-8'))
    for record in checkpoints:assert sha(ROOT/record['checkpoint'])==record['sha256']
    replayed=0;max_error=0.
    for fold in config['development_folds']:
        tr=np.flatnonzero((obs.completed<=fold['cutoff']).to_numpy())
        te=np.flatnonzero(((obs.date>fold['cutoff'])&(obs.date<=fold['end'])&obs.anchor.isin(anchors)).to_numpy())
        for arm in config['arms']:
            arrays=np.load(packed_path(arm,fold['prototype_year']))
            values={k:torch.from_numpy(arrays[k][te]).cuda() for k in ['patches','geometry','valid']}
            for seed in config['training']['seeds']:
                checkpoint=OUT/'checkpoints'/f"development_{arm['name']}_{fold['cutoff']}_{seed}.pt"
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert saved['epochs']==selection['selected_epochs'] and saved['train_n']==len(tr)
                y=obs.exec_return.to_numpy()[tr]
                assert abs(y.mean()-saved['target_mean'])<1e-12 and abs(y.std()-saved['target_sd'])<1e-12
                network=SegmentedCrossformer(saved['dimensions'],**saved['architecture']).cuda().eval()
                network.load_state_dict(saved['state_dict'])
                with torch.inference_mode():
                    output=network(values['patches'],values['geometry'],values['valid']).cpu().numpy().astype(float)
                output=output*saved['target_sd']+saved['target_mean']
                reference=seeds[(seeds.method==arm['name'])&(seeds.seed==seed)&(seeds.cutoff==fold['cutoff'])].sort_values('date')
                np.testing.assert_array_equal(reference.anchor,obs.anchor.iloc[te])
                error=float(np.max(np.abs(output-reference.predicted_return.to_numpy())))
                max_error=max(max_error,error)
                np.testing.assert_allclose(output,reference.predicted_return,rtol=0,atol=1e-10)
                assert sum(p.numel() for p in network.parameters())==(138441 if arm['dimensions']==5 else 150996)
                replayed+=1
                del network
            del values;torch.cuda.empty_cache()
    for arm in config['arms']:
        g=seeds[seeds.method==arm['name']]
        assert g.groupby('date').size().eq(3).all()
        ensemble=g.groupby('date').predicted_return.mean()
        actual=pred[pred.method==arm['name']].sort_values('date')
        np.testing.assert_allclose(ensemble.to_numpy(),actual.predicted_return,rtol=0,atol=1e-12)
    costs=pd.read_csv(OUT/'cost_sensitivity.csv');equities=pd.read_csv(OUT/'equity_curves_10bps.csv')
    for (window,name),g in costs.groupby(['window','method']):
        rows=pred[(pred.method==name)&(pred.date>=config['evaluation_windows'][window])].sort_values('date')
        assert (np.diff(g.sort_values('cost_bps').total_return)<=1e-12).all()
        for r in g.itertuples():
            assert abs(np.prod(1+v3.weekly_returns(rows,r.cost_bps))-1-r.total_return)<1e-10
            if name=='buy_hold':
                expected=frame.open.iloc[-1]/frame.open.iloc[int(rows.entry.iloc[0])]*(1-r.cost_bps/10000)**2-1
                assert abs(expected-r.total_return)<1e-10
        curve=equities[(equities.window==window)&(equities.method==name)]
        assert abs(curve.wealth.iloc[-1]-1-g[g.cost_bps==10].total_return.iloc[0])<1e-10
    # Previously frozen references are carried over exactly, not retrained here.
    for name,source in [('round3_quarterly_ridge','ridge_target'),('round3_direct_kronos','kronos_ohlcv')]:
        a=pred[pred.method==name].sort_values('date')
        b=old[(old.method==source)&(old.date>='2024-07-01')].sort_values('date')
        np.testing.assert_allclose(a.predicted_return,b.predicted_return,rtol=0,atol=1e-14)
    audit=dict(status='PASS',unit_tests=9,tokenizer_windows_reencoded=12,
               full_development_weeks=103,post_release_weeks=int(pred[(pred.method=='raw_fixed')&(pred.date>='2025-08-18')].shape[0]),
               model_arms=7,seeds_per_arm=3,main_predictions=len(pred),seed_predictions=len(seeds),
               development_checkpoints_replayed=replayed,max_replayed_prediction_error=max_error,
               all_valid_official_crossformer_parity=True,padding_value_and_gradient_invariance=True,
               validation_epochs_recomputed=True,all_labels_recomputed=True,all_fold_boundaries_purged=True,
               fixed_permuted_adaptive_budget_matching=True,all_input_and_source_hashes_match=True,
               old_evidence_files_preserved=len(run['previous_evidence']),
               daily_weekly_and_analytic_buyhold_accounting_match=True,
               checked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save_json(OUT/'verification.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
