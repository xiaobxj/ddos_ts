"""Complete-budget scoring; history perturbations never enter candidate selection."""
from common6 import *


def metric(g):
    y = g.actual.to_numpy(float)
    p = g.predicted_return.to_numpy(float)
    result = stats(p, y)
    positive = y > 0
    result.update(n=len(g), balanced_accuracy=float(((p[positive] > 0).mean() +
                  (p[~positive] <= 0).mean()) / 2), predicted_up_fraction=float((p > 0).mean()))
    if 'training_mean' in g:
        result['mse_skill_vs_training_mean'] = 1 - result['mse'] / float(np.mean((g.training_mean - y) ** 2))
    if 'cutoff' in g:
        centered = p - g.groupby('cutoff').predicted_return.transform('mean').to_numpy()
        total = float(np.sum((p - p.mean()) ** 2))
        result['within_fold_forecast_std'] = float(np.sqrt(np.mean(centered ** 2)))
        result['between_fold_variance_share'] = float(1-np.sum(centered**2)/total) if total else None
    return result


def bootstrap_indices(n):
    rng = np.random.default_rng(20260910)
    starts = rng.integers(n, size=(10000, int(np.ceil(n / 8))))
    return ((starts[:, :, None] + np.arange(8)) % n).reshape(10000, -1)[:, :n]


def difference(values, ids):
    values = np.asarray(values, float)
    mean = values.mean()
    boot = values[ids].mean(axis=1)
    centered = (values - mean)[ids].mean(axis=1)
    return dict(difference=float(mean), ci95_low=float(np.quantile(boot, .025)),
                ci95_high=float(np.quantile(boot, .975)),
                p=float((1 + np.sum(np.abs(centered) >= abs(mean))) / (len(ids) + 1)))


def holm(ps):
    order = np.argsort(ps)
    result = np.empty(len(ps))
    previous = 0.
    for i, j in enumerate(order):
        previous = max(previous, min(1., ps[j] * (len(ps) - i)))
        result[j] = previous
    return result.tolist()


def score_main():
    manifest = json.loads((OUT / 'main_manifest.json').read_text(encoding='utf-8'))
    assert manifest.get('finished_utc') and manifest['fits'] == 45
    seed = pd.read_csv(OUT / 'main_seed_predictions.csv')
    bases = pd.read_csv(OUT / 'main_baselines.csv')
    ensemble = seed.groupby(['arm', 'epoch', 'date'], sort=False).agg(
        predicted_return=('predicted_return', 'mean'), actual=('actual', 'first'),
        cutoff=('cutoff', 'first')).reset_index()
    ensemble = ensemble.merge(bases[['date', 'training_mean']], on='date', validate='many_to_one')
    rows = []
    years = []
    seeds = []
    for (arm, epoch), g in ensemble.groupby(['arm', 'epoch'], sort=False):
        assert len(g) == 141
        rows.append(dict(arm=arm, epoch=int(epoch), **metric(g)))
        for year, h in g.groupby(g.date.str[:4]):
            years.append(dict(arm=arm, epoch=int(epoch), year=int(year), **metric(h)))
    for (arm, epoch, number), g in seed.groupby(['arm', 'epoch', 'seed'], sort=False):
        g = g.merge(bases[['date', 'training_mean']], on='date', validate='one_to_one')
        seeds.append(dict(arm=arm, epoch=int(epoch), seed=int(number), **metric(g)))
    baselines = []
    for name in ['training_mean', 'zero_return']:
        g = bases.copy()
        g['predicted_return'] = g[name]
        baselines.append(dict(method=name, **metric(g)))
    order = {a['name']: i for i, a in enumerate(cfg()['arms'])}
    winner = min(rows, key=lambda r: (r['rmse'], r['epoch'], order[r['arm']]))
    save(OUT / 'selection.json', dict(selected=winner, baseline_metrics=baselines,
         candidates=len(rows), validation_weeks=len(bases), rule=cfg()['selection'],
         full_history_only=True, selected_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    pd.DataFrame(rows).to_csv(OUT / 'main_metrics.csv', index=False)
    pd.DataFrame(years).to_csv(OUT / 'main_yearly_metrics.csv', index=False)
    pd.DataFrame(seeds).to_csv(OUT / 'main_seed_metrics.csv', index=False)
    ensemble.to_csv(OUT / 'main_ensemble_predictions.csv', index=False)
    pairs = []
    reference = ensemble[(ensemble.arm == 'baseline') & (ensemble.epoch == 10)].sort_values('date')
    ids = bootstrap_indices(len(reference))
    for arm in cfg()['primary_comparisons']['arms']:
        g = ensemble[(ensemble.arm == arm) & (ensemble.epoch == 10)].sort_values('date')
        assert g.date.tolist() == reference.date.tolist()
        y = g.actual.to_numpy()
        a = (g.predicted_return.to_numpy() - y) ** 2
        b = (reference.predicted_return.to_numpy() - y) ** 2
        delta = np.sqrt(a[ids].mean(axis=1)) - np.sqrt(b[ids].mean(axis=1))
        pairs.append(dict(arm=arm, reference='baseline', epoch=10, mse=difference(a-b, ids),
                          rmse=dict(difference=float(np.sqrt(a.mean()) - np.sqrt(b.mean())),
                                    ci95_low=float(np.quantile(delta, .025)), ci95_high=float(np.quantile(delta, .975))),
                          accuracy=difference(((g.predicted_return > 0) == (g.actual > 0)).to_numpy(float) -
                                              ((reference.predicted_return > 0) == (reference.actual > 0)).to_numpy(float), ids)))
    for row, p in zip(pairs, holm([r['mse']['p'] for r in pairs])):
        row['mse']['holm_adjusted_p'] = p
    save(OUT / 'primary_comparisons.json', pairs)
    old = pd.read_csv(V5 / 'results/validation_seed_predictions.csv')
    old = old[old.variant == 'joint_ohlcv']
    current = seed[seed.arm == 'baseline']
    parity = old.merge(current, on=['epoch', 'seed', 'cutoff', 'row_index', 'date'],
                       suffixes=('_old', '_new'), validate='one_to_one')
    assert len(parity) == 1692
    maximum = float(np.max(np.abs(parity.predicted_return_old - parity.predicted_return_new)))
    np.testing.assert_allclose(parity.predicted_return_old, parity.predicted_return_new, rtol=0, atol=1e-10)
    np.testing.assert_allclose(parity.actual_old, parity.actual_new, rtol=0, atol=1e-12)
    prior4 = json.loads((V5 / 'results/frozen_round4_reference.json').read_text(encoding='utf-8'))
    save(OUT / 'previous_reference.json', dict(round5_baseline_rows_matched=len(parity),
         maximum_round5_prediction_difference=maximum, round4_reference=prior4,
         winner_minus_round4_rmse=winner['rmse']-prior4['rmse'],
         winner_minus_round5_joint10_rmse=winner['rmse']-metric(reference)['rmse']))
    print(pd.DataFrame(rows)[['arm', 'epoch', 'accuracy', 'rmse', 'mse_skill_vs_training_mean']].to_string(index=False))
    print('Selected: ' + json.dumps(winner))


def score_blocks():
    manifest = json.loads((OUT / 'blocks_manifest.json').read_text(encoding='utf-8'))
    assert manifest.get('finished_utc') and manifest['fits'] == 45
    main = pd.read_csv(OUT / 'main_seed_predictions.csv')
    full = main[(main.seed == 20260910) & (main.epoch == 10)]
    blocks = pd.read_csv(OUT / 'blocks_seed_predictions.csv')
    joined = pd.concat([full, blocks], ignore_index=True)
    bases = pd.concat([pd.read_csv(OUT / 'main_baselines.csv'),
                       pd.read_csv(OUT / 'blocks_baselines.csv')], ignore_index=True)
    joined = joined.merge(bases[['history', 'date', 'training_mean']],
                          on=['history', 'date'], validate='many_to_one')
    metrics = []
    yearly = []
    shifts = []
    for (arm, history), g in joined.groupby(['arm', 'history'], sort=False):
        g = g.sort_values('date')
        assert len(g) == 141
        reference = joined[(joined.history == history) & (joined.arm == 'baseline')].sort_values('date')
        assert reference.date.tolist() == g.date.tolist()
        measured = metric(g)
        measured['mse_difference_vs_matched_baseline'] = measured['mse'] - metric(reference)['mse']
        metrics.append(dict(arm=arm, history=history, seed=20260910, epoch=10, **measured))
        for year, h in g.groupby(g.date.str[:4]):
            yearly.append(dict(arm=arm, history=history, year=int(year), **metric(h)))
        if history != 'full':
            original = joined[(joined.history == 'full') & (joined.arm == arm)].sort_values('date')
            assert original.date.tolist() == g.date.tolist()
            p = g.predicted_return.to_numpy()
            p0 = original.predicted_return.to_numpy()
            mu = g.training_mean.to_numpy()
            mu0 = original.training_mean.to_numpy()
            shifts.append(dict(arm=arm, history=history,
                               rmse_difference_from_full=measured['rmse']-metric(original)['rmse'],
                               mse_difference_from_full=measured['mse']-metric(original)['mse'],
                               prediction_rms_shift=float(np.sqrt(np.mean((p-p0)**2))),
                               training_mean_rms_shift=float(np.sqrt(np.mean((mu-mu0)**2))),
                               prediction_rms_shift_after_removing_training_means=float(np.sqrt(np.mean(((p-mu)-(p0-mu0))**2))),
                               prediction_sign_disagreement=float(((p>0)!=(p0>0)).mean()),
                               changed_direction_weeks=int(((p>0)!=(p0>0)).sum())))
    table = pd.DataFrame(metrics)
    table['rank_within_history'] = table.groupby('history').rmse.rank(method='min').astype(int)
    table.to_csv(OUT / 'block_metrics.csv', index=False)
    pd.DataFrame(yearly).to_csv(OUT / 'block_yearly_metrics.csv', index=False)
    pd.DataFrame(shifts).to_csv(OUT / 'block_prediction_sensitivity.csv', index=False)
    joined.to_csv(OUT / 'block_comparison_predictions.csv', index=False)
    baselines = []
    for history, g in bases.groupby('history', sort=False):
        for name in ['training_mean', 'zero_return']:
            h = g.copy()
            h['predicted_return'] = h[name]
            baselines.append(dict(history=history, method=name, **metric(h)))
    save(OUT / 'block_baseline_metrics.json', baselines)
    stability = []
    for arm, g in table.groupby('arm', sort=False):
        perturbed = g[g.history != 'full']
        h = pd.DataFrame(shifts)
        h = h[h.arm == arm]
        original_gap = float(g.loc[g.history == 'full', 'mse_difference_vs_matched_baseline'].iloc[0])
        stability.append(dict(arm=arm, better_than_training_mean_histories=int((g.mse_skill_vs_training_mean > 0).sum()),
                              better_than_matched_baseline_histories=int((g.mse_difference_vs_matched_baseline < 0).sum()),
                              histories=4, best_rmse=float(g.rmse.min()), worst_rmse=float(g.rmse.max()),
                              best_rank=int(g.rank_within_history.min()), worst_rank=int(g.rank_within_history.max()),
                              full_history_mse_gap_vs_baseline=original_gap,
                              relative_advantage_sign_reversals=int(((perturbed.mse_difference_vs_matched_baseline < 0) !=
                                                                    (original_gap < 0)).sum()) if arm != 'baseline' else None,
                              maximum_prediction_rms_shift=float(h.prediction_rms_shift.max()),
                              maximum_direction_disagreement=float(h.prediction_sign_disagreement.max())))
    save(OUT / 'block_stability_summary.json', stability)
    # Training-only interpretation of the history deletion. This does not enter
    # candidate selection or change the frozen training/scaling behavior.
    obs = pd.read_csv(OUT / 'observation_table.csv')
    label_rows = []
    for fold in cfg()['folds']:
        histories, _ = fold_indices(obs, fold)
        full_mean = obs.exec_return.iloc[histories['full']].mean()
        for history, tr in histories.items():
            y = obs.exec_return.iloc[tr].to_numpy()
            adjacent = np.diff(obs.anchor.iloc[tr].to_numpy()) == 1
            label_rows.append(dict(cutoff=fold['cutoff'], history=history, training_n=len(tr),
                                   label_mean=float(y.mean()), label_std=float(y.std()),
                                   mean_change_from_full=float(y.mean()-full_mean),
                                   adjacent_training_pairs=int(adjacent.sum()),
                                   adjacent_label_correlation=float(np.corrcoef(y[:-1][adjacent], y[1:][adjacent])[0, 1])))
    pd.DataFrame(label_rows).to_csv(OUT / 'training_history_label_statistics.csv', index=False)
    print(table[['arm', 'history', 'accuracy', 'rmse', 'mse_skill_vs_training_mean',
                 'mse_difference_vs_matched_baseline', 'rank_within_history']].to_string(index=False))


def main():
    score_main()
    if (OUT / 'blocks_manifest.json').exists():
        run = json.loads((OUT / 'blocks_manifest.json').read_text(encoding='utf-8'))
        if run.get('finished_utc'):
            score_blocks()


if __name__ == '__main__':
    main()
