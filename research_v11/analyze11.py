"""Descriptive paired comparisons of training losses; no predictive scoring."""
from common11 import *

METRICS = ['return_mse', 'return_huber', 'auxiliary_mse', 'common_joint', 'linear_fraction']


def summarize_loss_draws(draws):
    result = []
    for (state, cutoff, seed, mode), group in draws.groupby(['state', 'cutoff', 'seed', 'mode'], sort=False):
        row = dict(state=state, cutoff=cutoff, seed=int(seed), mode=mode, draws=len(group))
        for key in METRICS:
            row[key] = float(group[key].mean())
            row[key + '_draw_sd'] = float(group[key].std(ddof=1)) if len(group) > 1 else 0.
        result.append(row)
    return pd.DataFrame(result)


def compare(losses, left, right, label):
    a = losses[losses.state.eq(left)].set_index(['cutoff','seed','mode'])
    b = losses[losses.state.eq(right)].set_index(['cutoff','seed','mode'])
    assert a.index.equals(b.index)
    rows = []
    for index, r in a.iterrows():
        s = b.loc[index]
        row = dict(comparison=label, left_state=left, right_state=right,
                   cutoff=index[0], seed=int(index[1]), mode=index[2])
        for key in METRICS:
            row[key+'_left'] = float(r[key])
            row[key+'_right'] = float(s[key])
            row[key+'_difference'] = float(r[key]-s[key])
            row[key+'_relative_change'] = float(r[key]/s[key]-1) if s[key] != 0 else None
        rows.append(row)
    return rows


def main():
    check_frozen()
    assert read(OUT / 'training_manifest.json').get('finished_utc')
    draws = pd.concat([pd.read_csv(OUT / 'archived_loss_draws.csv'), pd.read_csv(OUT / 'restart_loss_draws.csv')], ignore_index=True)
    assert len(draws) == 324
    losses = summarize_loss_draws(draws)
    losses.to_csv(OUT / 'training_loss_summary.csv', index=False)
    pairs = []
    for left,right,label in [('archived_huber','archived_mse','archived_huber_minus_mse'),
                             ('joint','archived_huber','joint_restart_minus_start'),
                             ('return_only','archived_huber','return_only_restart_minus_start'),
                             ('return_only','joint','return_only_minus_joint'),
                             ('joint','archived_mse','joint_restart_minus_archived_mse')]:
        pairs.extend(compare(losses, left, right, label))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / 'paired_training_comparisons.csv', index=False)
    summaries = []
    for (comparison,mode), g in pairs.groupby(['comparison','mode']):
        row = dict(comparison=comparison, mode=mode, pairs=len(g))
        for key in METRICS:
            row[key+'_left_mean'] = float(g[key+'_left'].mean())
            row[key+'_right_mean'] = float(g[key+'_right'].mean())
            row[key+'_difference_mean'] = float(g[key+'_difference'].mean())
            row[key+'_mean_relative_change'] = float(g[key+'_left'].mean()/g[key+'_right'].mean()-1)
            row[key+'_lower_count'] = int(g[key+'_difference'].lt(-1e-7).sum())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT / 'comparison_summary.csv', index=False)
    gradient = pd.read_csv(OUT / 'gradient_diagnostics.csv')
    gradient_summaries = []
    for (state,mode,group), g in gradient[gradient['mode'].isin(['eval','dropout_mean4'])].groupby(['state','mode','parameter_group']):
        gradient_summaries.append(dict(state=state, mode=mode, parameter_group=group, states=len(g),
            negative_cosine_count=int(g.cosine.lt(-1e-8).sum()),
            cosine_mean=float(g.cosine.mean()), cosine_min=float(g.cosine.min()), cosine_max=float(g.cosine.max()),
            weighted_auxiliary_norm_ratio_mean=float(g.weighted_auxiliary_norm_ratio.mean()),
            auxiliary_projection_fraction_mean=float(g.auxiliary_projection_fraction.mean()),
            auxiliary_projection_fraction_min=float(g.auxiliary_projection_fraction.min()),
            auxiliary_projection_fraction_max=float(g.auxiliary_projection_fraction.max()),
            full_joint_step_opposes_return_count=int(g.auxiliary_projection_fraction.lt(-1).sum())))
    pd.DataFrame(gradient_summaries).to_csv(OUT / 'gradient_summary.csv', index=False)
    result = dict(experiment=11, training_only=True, new_validation_predictions=0, new_validation_metrics=0,
                  archived_states=18, new_fits=18, fixed_restart_epochs=20, loss_passes=324,
                  monte_carlo_draws=288, gradient_passes=90, paired_training_comparisons=90,
                  comparison_summary=summaries, gradient_summary=gradient_summaries,
                  limitation='Training optimization mechanisms only; no convergence proof, predictive promotion or independent confirmation.')
    save(OUT / 'assessment.json', result)
    print(json.dumps(dict(status='ANALYZED', training_only=True, model_states=36, new_validation_metrics=0)))


if __name__ == '__main__':
    main()
