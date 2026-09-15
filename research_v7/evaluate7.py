"""Fixed-role multiseed scoring; no model/epoch/deletion selection."""
from common7 import *
import evaluate6 as old_statistics


def metric(g):
    result = old_statistics.metric(g)
    if 'reassigned_prediction' in g:
        changed = float(np.mean((g.reassigned_prediction-g.actual)**2))
        result.update(reassigned_mse=changed, reassignment_mse_increase=changed-result['mse'])
    return result


def load_predictions():
    frames = [pd.read_csv(OUT / 'reused_full_predictions.csv')]
    for arm in cfg()['training']['arms']:
        run = json.loads((OUT / f'{arm}_manifest.json').read_text(encoding='utf-8'))
        assert run.get('finished_utc') and run['fits'] == 27
        frames.append(pd.read_csv(OUT / f'{arm}_predictions.csv'))
    seed = pd.concat(frames, ignore_index=True)
    assert len(seed) == 6768 and not seed.duplicated(['role', 'history', 'seed', 'date']).any()
    assert seed.groupby(['role', 'history', 'date']).size().eq(3).all()
    obs = pd.read_csv(OUT / 'observation_table.csv')
    base = []
    for fold in cfg()['folds']:
        with np.load(CACHE / f'masks_{fold["cutoff"]}.npz') as masks:
            for history in cfg()['histories']:
                ymean = float(obs.exec_return.iloc[masks[history]].mean())
                for i in masks['testing']:
                    base.append(dict(history=history, cutoff=fold['cutoff'], date=obs.date.iloc[i],
                                     actual=float(obs.exec_return.iloc[i]), training_mean=ymean, zero_return=0.))
    baselines = pd.DataFrame(base)
    seed = seed.merge(baselines[['history', 'date', 'training_mean']], on=['history', 'date'], validate='many_to_one')
    ensemble = seed.groupby(['role', 'history', 'date'], sort=False).agg(
        predicted_return=('predicted_return', 'mean'), reassigned_prediction=('reassigned_prediction', 'mean'),
        actual=('actual', 'first'), cutoff=('cutoff', 'first'), training_mean=('training_mean', 'first')).reset_index()
    return seed, ensemble, baselines


def paired(candidate, reference, history, reference_name):
    a = candidate.sort_values('date'); b = reference.sort_values('date')
    assert a.date.tolist() == b.date.tolist()
    y = a.actual.to_numpy()
    x = (a.predicted_return.to_numpy()-y)**2
    z = (b.predicted_return.to_numpy()-y)**2
    ids = old_statistics.bootstrap_indices(len(a))
    delta = np.sqrt(x[ids].mean(axis=1))-np.sqrt(z[ids].mean(axis=1))
    return dict(history=history, candidate='candidate20', reference=reference_name, n=len(a),
                mse=old_statistics.difference(x-z, ids),
                rmse=dict(difference=float(np.sqrt(x.mean())-np.sqrt(z.mean())),
                          ci95_low=float(np.quantile(delta, .025)), ci95_high=float(np.quantile(delta, .975))),
                accuracy=old_statistics.difference(((a.predicted_return>0)==(a.actual>0)).to_numpy(float)-
                                                  ((b.predicted_return>0)==(b.actual>0)).to_numpy(float), ids))


def sensitivity(seed, ensemble):
    rows = []
    groups = [('ensemble', None, ensemble)] + [('seed', int(number), g) for number, g in seed.groupby('seed')]
    for level, number, frame in groups:
        for name, g in frame.groupby('role', sort=False):
            full = g[g.history == 'full'].sort_values('date')
            p0 = full.predicted_return.to_numpy(); mu0 = full.training_mean.to_numpy()
            for history in cfg()['histories'][1:]:
                changed = g[g.history == history].sort_values('date')
                assert changed.date.tolist() == full.date.tolist()
                p = changed.predicted_return.to_numpy(); mu = changed.training_mean.to_numpy()
                rows.append(dict(level=level, seed=number, role=name, history=history,
                                 prediction_rms_shift=float(np.sqrt(np.mean((p-p0)**2))),
                                 training_mean_rms_shift=float(np.sqrt(np.mean((mu-mu0)**2))),
                                 mean_removed_prediction_rms_shift=float(np.sqrt(np.mean(((p-mu)-(p0-mu0))**2))),
                                 direction_disagreement=float(((p>0)!=(p0>0)).mean()),
                                 changed_direction_weeks=int(((p>0)!=(p0>0)).sum()),
                                 mse_change_from_full=metric(changed)['mse']-metric(full)['mse']))
    return pd.DataFrame(rows)


def main():
    seed, ensemble, bases = load_predictions()
    rows = []; years = []; seeds = []; base_metrics = []; comparison_rows = []
    for (name, history), g in ensemble.groupby(['role', 'history'], sort=False):
        assert len(g) == 141
        rows.append(dict(role=name, history=history, **metric(g)))
        for year, h in g.groupby(g.date.str[:4]):
            years.append(dict(role=name, history=history, year=int(year), **metric(h)))
    for (name, history, number), g in seed.groupby(['role', 'history', 'seed'], sort=False):
        seeds.append(dict(role=name, history=history, seed=int(number), **metric(g)))
    for history, g in bases.groupby('history', sort=False):
        for name in ['training_mean', 'zero_return']:
            h = g.copy(); h['predicted_return'] = h[name]
            base_metrics.append(dict(role=name, history=history, **metric(h)))
    metrics = pd.DataFrame(rows); seed_metrics = pd.DataFrame(seeds)
    for history in cfg()['histories']:
        candidate = metrics[(metrics.role=='candidate20') & (metrics.history==history)].iloc[0]
        b10 = metrics[(metrics.role=='baseline10') & (metrics.history==history)].iloc[0]
        b20 = metrics[(metrics.role=='baseline20') & (metrics.history==history)].iloc[0]
        individual = seed_metrics[(seed_metrics.role=='candidate20') & (seed_metrics.history==history)]
        mean = next(r for r in base_metrics if r['history']==history and r['role']=='training_mean')
        comparison_rows.append(dict(history=history, candidate_accuracy=float(candidate.accuracy),
                                    candidate_rmse=float(candidate.rmse), baseline10_rmse=float(b10.rmse),
                                    baseline20_rmse=float(b20.rmse), training_mean_rmse=mean['rmse'],
                                    mse_skill_vs_mean=float(candidate.mse_skill_vs_training_mean),
                                    mse_skill_vs_baseline10=1-float(candidate.mse/b10.mse),
                                    mse_skill_vs_baseline20=1-float(candidate.mse/b20.mse),
                                    candidate_seeds_better_than_mean=int((individual.mse_skill_vs_training_mean>0).sum())))
    comparisons = pd.DataFrame(comparison_rows)
    deleted = comparisons[comparisons.history!='full']
    flags = dict(absolute_advantage=bool((deleted.mse_skill_vs_mean>0).all()),
                 relative_advantage=bool((deleted.mse_skill_vs_baseline10>0).all()),
                 seed_consistency=bool((deleted.candidate_seeds_better_than_mean>=2).all()))
    primary = []
    for history in cfg()['histories'][1:]:
        candidate = ensemble[(ensemble.role=='candidate20') & (ensemble.history==history)]
        reference = ensemble[(ensemble.role=='baseline10') & (ensemble.history==history)]
        primary.append(paired(candidate, reference, history, 'baseline10'))
        mean = bases[bases.history==history].copy(); mean['predicted_return'] = mean.training_mean
        primary.append(paired(candidate, mean, history, 'training_mean'))
    for row, p in zip(primary, old_statistics.holm([r['mse']['p'] for r in primary])):
        row['mse']['holm_adjusted_p'] = p
    save(OUT / 'primary_comparisons.json', primary)
    save(OUT / 'stability_assessment.json', dict(candidate='combined, epoch20', flags=flags,
         consistency_screen_pass=all(flags.values()), rules=cfg()['stability_flags'],
         deleted_histories_better_than_mean=int((deleted.mse_skill_vs_mean>0).sum()),
         deleted_histories_better_than_baseline10=int((deleted.mse_skill_vs_baseline10>0).sum()),
         primary_comparisons_with_holm_p_below_0_05=sum(r['mse']['holm_adjusted_p']<.05 for r in primary),
         primary_significant_candidate_improvements=sum(r['mse']['holm_adjusted_p']<.05 and r['mse']['difference']<0 for r in primary),
         no_candidate_selection=True, historical_reuse=True, scored_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    save(OUT / 'baseline_metrics.json', base_metrics)
    seed.to_csv(OUT / 'all_seed_predictions.csv', index=False)
    ensemble.to_csv(OUT / 'ensemble_predictions.csv', index=False)
    metrics.to_csv(OUT / 'ensemble_metrics.csv', index=False)
    seed_metrics.to_csv(OUT / 'seed_metrics.csv', index=False)
    pd.DataFrame(years).to_csv(OUT / 'yearly_metrics.csv', index=False)
    comparisons.to_csv(OUT / 'candidate_comparisons.csv', index=False)
    sensitivity(seed, ensemble).to_csv(OUT / 'prediction_sensitivity.csv', index=False)
    bases.to_csv(OUT / 'baselines.csv', index=False)
    print(comparisons.to_string(index=False))
    print(json.dumps(dict(flags=flags, consistency_screen_pass=all(flags.values())), indent=2))


if __name__ == '__main__':
    main()
