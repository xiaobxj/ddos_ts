"""Independent mask/scale reconstruction and all-checkpoint prediction replay."""
from models6 import *
import subprocess


def reconstruct(obs, fold, history):
    full = np.where(obs.joint_completed.to_numpy() <= fold['cutoff'])[0]
    testing = np.where((obs.date.to_numpy() > fold['cutoff']) &
                       (obs.date.to_numpy() <= fold['end']) & (obs.weekday.to_numpy() == 4) &
                       (obs.joint_completed.to_numpy() <= '2020-12-31'))[0]
    if history == 'full':
        return full, full, testing
    block = {'drop_early': 0, 'drop_middle': 2, 'drop_recent': 4}[history]
    lengths = np.full(5, len(full) // 5)
    lengths[:len(full) % 5] += 1
    edges = np.r_[0, np.cumsum(lengths)]
    keep = np.ones(len(full), bool)
    keep[edges[block]:edges[block+1]] = False
    return full[keep], full, testing


def replay_phase(phase, obs, targets):
    predictions = pd.read_csv(OUT / f'{phase}_seed_predictions.csv')
    checkpoints = json.loads((OUT / f'{phase}_checkpoint_manifest.json').read_text(encoding='utf-8'))
    curves = pd.read_csv(OUT / f'{phase}_training_curves.csv')
    baselines = pd.read_csv(OUT / f'{phase}_baselines.csv')
    expected_cp = 180 if phase == 'main' else 45
    assert len(checkpoints) == expected_cp
    assert len(predictions) == (8460 if phase == 'main' else 2115)
    assert not predictions.duplicated(['arm', 'history', 'epoch', 'seed', 'date']).any()
    replayed = 0
    maximum = 0.
    for fold in cfg()['folds']:
        histories = ['full'] if phase == 'main' else cfg()['training_block_sensitivity']['masks']
        for history in histories:
            tr, full, te = reconstruct(obs, fold, history)
            assert obs.joint_completed.iloc[tr].max() <= fold['cutoff'] < obs.date.iloc[te].min()
            with np.load(CACHE / f'masks_{fold["cutoff"]}.npz') as masks:
                np.testing.assert_array_equal(masks[history], tr)
                np.testing.assert_array_equal(masks['testing'], te)
            values = batch_tensors(te)
            baseline = baselines[(baselines.cutoff == fold['cutoff']) & (baselines.history == history)]
            np.testing.assert_allclose(baseline.training_mean, targets['returns'][tr].mean(), rtol=0, atol=1e-12)
            c = curves[(curves.cutoff == fold['cutoff']) & (curves.history == history)]
            assert c.presentations.eq(len(full)).all()
            assert c.train_n.eq(len(tr)).all()
            assert c.optimizer_steps.eq(int(np.ceil(len(full)/128))).all()
            expected_epochs = 20 if phase == 'main' else 10
            expected_seeds = 3 if phase == 'main' else 1
            assert len(c) == 5 * expected_seeds * expected_epochs
            selected = [r for r in checkpoints if r['cutoff'] == fold['cutoff'] and r['history'] == history]
            for r in selected:
                path = ROOT / r['file']
                assert sha(path) == r['sha256'], path
                saved = torch.load(path, map_location='cpu', weights_only=True)
                for key in ['arm', 'history', 'seed', 'epoch', 'cutoff']:
                    assert saved[key] == r[key]
                assert saved['protocol_sha256'] == sha(ROOT / 'protocol.json')
                assert saved['train_n'] == len(tr) and saved['full_train_n'] == len(full)
                for key in ['returns', 'auxiliary']:
                    y = targets[key][tr].astype(float)
                    np.testing.assert_allclose(saved['scales'][key+'_mean'], y.mean(axis=0), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(saved['scales'][key+'_sd'], np.maximum(y.std(axis=0), 1e-6), rtol=0, atol=1e-12)
                network = make_model(r['arm'], r['seed']).eval()
                network.load_state_dict(saved['state_dict'])
                with torch.inference_mode():
                    output, _ = predict(network, values, False)
                estimate = output.cpu().numpy().astype(float) * saved['scales']['returns_sd'] + saved['scales']['returns_mean']
                reference = predictions[(predictions.arm == r['arm']) & (predictions.history == history) &
                                        (predictions.seed == r['seed']) & (predictions.epoch == r['epoch']) &
                                        (predictions.cutoff == r['cutoff'])]
                np.testing.assert_array_equal(reference.row_index, te)
                np.testing.assert_allclose(reference.actual, targets['returns'][te], rtol=0, atol=1e-12)
                error = float(np.max(np.abs(estimate - reference.predicted_return)))
                maximum = max(maximum, error)
                np.testing.assert_allclose(estimate, reference.predicted_return, rtol=0, atol=1e-10)
                replayed += 1
                del network
                torch.cuda.empty_cache()
            del values
            print(f'Checkpoint replay: {phase}, {fold["cutoff"]}, {history}, {replayed}/{expected_cp}', flush=True)
    assert replayed == expected_cp
    return dict(checkpoints_replayed=replayed, maximum_prediction_error=maximum, prediction_rows=len(predictions))


def check_scores():
    import evaluate6 as scoring
    seeds = pd.read_csv(OUT / 'main_seed_predictions.csv')
    means = seeds.groupby(['arm', 'epoch', 'date']).predicted_return.mean()
    ensemble = pd.read_csv(OUT / 'main_ensemble_predictions.csv').set_index(['arm', 'epoch', 'date']).sort_index()
    np.testing.assert_allclose(ensemble.predicted_return, means.sort_index(), rtol=0, atol=1e-12)
    assert seeds.groupby(['arm', 'epoch', 'date']).size().eq(3).all()
    metrics = pd.read_csv(OUT / 'main_metrics.csv')
    measured = []
    for r in metrics.itertuples():
        g = ensemble.reset_index()
        g = g[(g.arm == r.arm) & (g.epoch == r.epoch)]
        result = scoring.metric(g)
        for key, value in result.items():
            if value is not None:
                assert abs(value-getattr(r, key)) < 1e-12, (r.arm, r.epoch, key)
        measured.append(dict(arm=r.arm, epoch=int(r.epoch), **result))
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    order = {a['name']: i for i, a in enumerate(cfg()['arms'])}
    winner = min(measured, key=lambda r: (r['rmse'], r['epoch'], order[r['arm']]))
    assert (winner['arm'], winner['epoch']) == (selection['selected']['arm'], selection['selected']['epoch'])
    block = pd.read_csv(OUT / 'block_comparison_predictions.csv')
    for r in pd.read_csv(OUT / 'block_metrics.csv').itertuples():
        g = block[(block.arm == r.arm) & (block.history == r.history)]
        result = scoring.metric(g)
        for key, value in result.items():
            if value is not None:
                assert abs(value-getattr(r, key)) < 1e-12, (r.arm, r.history, key)
    prior = pd.read_csv(V5 / 'results/validation_seed_predictions.csv')
    prior = prior[prior.variant == 'joint_ohlcv']
    new = seeds[seeds.arm == 'baseline']
    matched = prior.merge(new, on=['epoch', 'seed', 'date', 'cutoff'], suffixes=('_old', '_new'), validate='one_to_one')
    assert len(matched) == 1692
    np.testing.assert_allclose(matched.predicted_return_old, matched.predicted_return_new, rtol=0, atol=1e-10)
    return dict(main_candidates=len(metrics), weekly_dates=len(ensemble.index.get_level_values('date').unique()),
                round5_baseline_predictions_matched=len(matched),
                round5_maximum_prediction_difference=float(np.max(np.abs(matched.predicted_return_old-matched.predicted_return_new))),
                selected_arm=winner['arm'], selected_epoch=winner['epoch'])


def main():
    initialize()
    start = time.time()
    test = subprocess.run([sys.executable, str(ROOT / 'test_contracts6.py')], capture_output=True,
                          text=True, encoding='utf-8', errors='replace')
    (OUT / 'contract_test_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    assert test.returncode == 0, test.stdout+test.stderr
    prior = old_evidence()
    for phase in ['preparation', 'main', 'blocks']:
        run = json.loads((OUT / f'{phase}_manifest.json').read_text(encoding='utf-8'))
        assert run.get('finished_utc') and run['protocol_sha256'] == sha(ROOT / 'protocol.json')
        assert run['source_sha256'] == source_hashes() and run['input_sha256'] == input_hashes()
        assert run['old_evidence'] == prior
        if phase != 'preparation':
            assert run['preparation_manifest_sha256'] == sha(OUT / 'preparation_manifest.json')
    prep = json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))
    for name, digest in prep['cache_sha256'].items():
        assert sha(ROOT / name) == digest
    assert sha(OUT / 'observation_table.csv') == prep['observation_table_sha256'] == sha(V5 / 'results/observation_table.csv')
    obs = pd.read_csv(OUT / 'observation_table.csv')
    targets = dict(np.load(V5 / 'cache/targets.npz'))
    main_replay = replay_phase('main', obs, targets)
    blocks_replay = replay_phase('blocks', obs, targets)
    scores = check_scores()
    result = dict(status='PASS', contract_tests=4, observations=len(obs), main=main_replay, blocks=blocks_replay,
                  **scores, all_input_and_source_hashes_match=True, all_training_labels_mature=True,
                  all_retained_training_scales_recomputed=True, all_update_budgets_match=True,
                  prior_evidence_files_preserved=len(prior), total_fits=90, total_checkpoints_replayed=225,
                  elapsed_seconds=time.time()-start, checked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
