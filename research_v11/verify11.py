"""Replay all states and fixed draws, original budgets and paired training conclusions."""
from common11 import *


def main():
    legacy.initialize()
    prep = check_frozen()
    assert old_evidence() == prep['old_evidence']
    start = time.time()
    refs = read(OUT / 'archived_models.json') + read(OUT / 'restart_models.json')
    assert len(refs) == 36
    draws = pd.concat([pd.read_csv(OUT / 'archived_loss_draws.csv'), pd.read_csv(OUT / 'restart_loss_draws.csv')], ignore_index=True)
    curves = pd.read_csv(OUT / 'restart_training_curves.csv')
    monitors = pd.read_csv(OUT / 'restart_training_monitors.csv')
    gradient_rows = pd.read_csv(OUT / 'gradient_diagnostics.csv')
    rows = pd.read_csv(OUT / 'training_rows.csv')
    assert len(curves) == 360 and len(monitors) == 90 and len(draws) == 324
    max_output, max_loss, max_gradient = 0., 0., 0.
    orders = 0
    layout = read(OUT / 'parameter_layout.json')
    for number, ref in enumerate(refs, 1):
        model, saved = load_model(ref)
        values, labels, tr, scales = load_training(ref['cutoff'])
        subset = rows[rows.cutoff.eq(ref['cutoff'])]
        np.testing.assert_array_equal(tr, subset.row_index.to_numpy())
        assert subset.joint_completed.le(ref['cutoff']).all() and saved['scales'] == scales
        current, prediction = loss_audit(model, values, labels)
        previous = draws[(draws.state.eq(ref['state'])) & draws.cutoff.eq(ref['cutoff']) & draws.seed.eq(ref['seed'])]
        assert len(previous) == 9
        for r in current:
            old = previous[(previous['mode']==r['mode']) & previous.draw.eq(r['draw'])].iloc[0]
            for key in ['return_mse','return_huber','auxiliary_mse','common_joint','linear_fraction']:
                delta = abs(old[key]-r[key]); max_loss = max(max_loss, delta)
                assert delta < 1e-11, (reference_id(ref), key, delta)
        with np.load(OUT / 'training_predictions' / (reference_id(ref)+'.npz')) as previous_outputs:
            np.testing.assert_array_equal(previous_outputs['row_index'], tr)
            for key, v in prediction.items():
                delta = float(np.max(np.abs(v-previous_outputs[key])))
                max_output = max(max_output,delta)
                assert delta < 1e-11
        if ref['state'].startswith('archived_'):
            with np.load(OUT / 'gradient_vectors' / (reference_id(ref)+'.npz')) as vectors:
                for draw in range(-1,4):
                    gr, ga = gradient_pass(model, values, labels, draw>=0, 20261100+draw if draw>=0 else None)
                    for x,y in [(gr,vectors['returns'][draw+1]),(ga,vectors['auxiliary'][draw+1])]:
                        delta = float(np.max(np.abs(x-y)))
                        max_gradient = max(max_gradient,delta)
                        assert delta < 1e-10
                cases = [(vectors['returns'][i],vectors['auxiliary'][i],'eval' if i==0 else 'dropout',i-1) for i in range(5)]
                cases.append((vectors['returns'][1:].mean(axis=0),vectors['auxiliary'][1:].mean(axis=0),'dropout_mean4',-2))
                for gr,ga,mode,draw in cases:
                    for stats in gradient_stats(gr,ga,layout):
                        g = gradient_rows[gradient_rows.state.eq(ref['state']) & gradient_rows.cutoff.eq(ref['cutoff']) &
                            gradient_rows.seed.eq(ref['seed']) & gradient_rows['mode'].eq(mode) & gradient_rows.draw.eq(draw) &
                            gradient_rows.parameter_group.eq(stats['parameter_group'])].iloc[0]
                        for key, value in stats.items():
                            if key != 'parameter_group':
                                assert abs(g[key]-value) < 1e-10
        else:
            assert saved['restart_epoch']==20 and saved['original_epoch']==20 and saved['training_only']
            assert saved['protocol_sha256']==sha(ROOT/'protocol.json')
            assert saved['final_state_sha256']==state_hash(model)==ref['final_state_sha256']
            group = curves[curves.state.eq(ref['state']) & curves.cutoff.eq(ref['cutoff']) & curves.seed.eq(ref['seed'])].sort_values('epoch')
            assert group.epoch.tolist() == list(range(1,21))
            rng = np.random.default_rng(ref['seed'] + 110000)
            for r in group.itertuples():
                order = rng.permutation(len(tr))
                assert r.training_order_sha256==array_hash(tr[order])
                assert r.batch_boundaries_sha256==array_hash(np.r_[np.arange(0,len(tr),128),len(tr)])
                assert r.train_n==r.presentations==len(tr) and r.optimizer_steps==(len(tr)+127)//128
                assert r.learning_rate==.0001
                orders += 1
            assert saved['numpy_rng_state']==rng.bit_generator.state
            optimizer = saved['optimizer_state_dict']
            assert all(int(s['step'])==20*((len(tr)+127)//128) for s in optimizer['state'].values())
            assert optimizer['param_groups'][0]['lr']==.0001 and optimizer['param_groups'][0]['weight_decay']==.1
            pair = next(r for r in refs if r['cutoff']==ref['cutoff'] and r['seed']==ref['seed'] and
                        r['state']==('return_only' if ref['state']=='joint' else 'joint'))
            assert pair['initial_state_sha256']==ref['initial_state_sha256']
            base = next(r for r in refs if r['cutoff']==ref['cutoff'] and r['seed']==ref['seed'] and r['state']=='archived_huber')
            starting, _ = load_model(base)
            assert ref['initial_state_sha256']==state_hash(starting)
            del starting
            m = monitors[monitors.state.eq(ref['state']) & monitors.cutoff.eq(ref['cutoff']) & monitors.seed.eq(ref['seed'])]
            assert m.epoch.tolist()==[0,5,10,15,20]
            initial = draws[draws.state.eq('archived_huber') & draws.cutoff.eq(ref['cutoff']) & draws.seed.eq(ref['seed']) & draws['mode'].eq('eval')].iloc[0]
            for key in ['return_mse','return_huber','auxiliary_mse','common_joint','linear_fraction']:
                assert abs(m[m.epoch.eq(0)].iloc[0][key]-initial[key])<1e-11
                assert abs(m[m.epoch.eq(20)].iloc[0][key]-current[0][key])<1e-11
        print(f'Training-only replay {number}/36; {time.time()-start:.1f}s', flush=True)
        del model, values, labels
        torch.cuda.empty_cache()
    assert orders==360
    summary = pd.read_csv(OUT/'training_loss_summary.csv')
    for r in summary.to_dict('records'):
        g = draws[draws.state.eq(r['state']) & draws.cutoff.eq(r['cutoff']) & draws.seed.eq(r['seed']) & draws['mode'].eq(r['mode'])]
        assert r['draws']==len(g)
        for key in ['return_mse','return_huber','auxiliary_mse','common_joint','linear_fraction']:
            assert abs(r[key]-g[key].mean())<1e-11
            sd = g[key].std(ddof=1) if len(g)>1 else 0.
            assert abs(r[key+'_draw_sd']-sd)<1e-11
    pairs = pd.read_csv(OUT/'paired_training_comparisons.csv')
    assert len(pairs)==90
    for r in pairs.to_dict('records'):
        mask = summary.cutoff.eq(r['cutoff']) & summary.seed.eq(r['seed']) & summary['mode'].eq(r['mode'])
        left = summary[mask & summary.state.eq(r['left_state'])].iloc[0]
        right = summary[mask & summary.state.eq(r['right_state'])].iloc[0]
        for key in ['return_mse','return_huber','auxiliary_mse','common_joint','linear_fraction']:
            assert abs(r[key+'_left']-left[key])<1e-11 and abs(r[key+'_right']-right[key])<1e-11
            assert abs(r[key+'_difference']-(left[key]-right[key]))<1e-11
    comparisons = pd.read_csv(OUT/'comparison_summary.csv')
    for r in comparisons.to_dict('records'):
        g = pairs[pairs.comparison.eq(r['comparison']) & pairs['mode'].eq(r['mode'])]
        assert len(g)==r['pairs']==9
        for key in ['return_mse','return_huber','auxiliary_mse','common_joint','linear_fraction']:
            assert r[key+'_lower_count']==int(g[key+'_difference'].lt(-1e-7).sum())
            assert abs(r[key+'_difference_mean']-g[key+'_difference'].mean())<1e-11
            assert abs(r[key+'_mean_relative_change']-(g[key+'_left'].mean()/g[key+'_right'].mean()-1))<1e-11
    check_frozen()
    assert old_evidence()==prep['old_evidence']
    result = dict(status='PASS', model_states_replayed=36, eval_loss_passes=36, dropout_loss_draws_replayed=288,
                  gradient_passes_replayed=90, maximum_output_replay_error=max_output,
                  maximum_loss_replay_error=max_loss, maximum_gradient_replay_error=max_gradient,
                  epoch_orders_and_boundaries_recomputed=orders, paired_comparisons_recomputed=90,
                  previous_files_preserved=len(prep['old_evidence']), new_validation_predictions=0,
                  elapsed_seconds=time.time()-start, completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
