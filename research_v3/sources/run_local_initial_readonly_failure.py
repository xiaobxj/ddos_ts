"""Small nonlinear baseline and fixed-parameter cross-index feature ablation."""
from common import *
import platform
import time
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


def hgb_predict(x, y, xt, config):
    model = HistGradientBoostingRegressor(**cfg()['hgb_common'], **config)
    model.fit(x, y)
    assert model.n_iter_ == config['max_iter'] and not model.do_early_stopping_
    return model.predict(xt)


def main():
    OUT.mkdir(exist_ok=True)
    protocol = cfg()
    started = time.time()
    manifest = dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    protocol_sha256=sha(ROOT / 'protocol.json'),
                    source_sha256={name: sha(ROOT / name) for name in ['common.py', 'run_local.py']},
                    previous_evidence=previous_evidence(), executable=sys.executable,
                    python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)
    save_json(OUT / 'local_run_manifest.json', manifest)
    frame = pd.read_csv(V1 / 'data/1_000300.csv')
    obs = observation_table(frame)
    panels = [pd.read_csv(V1 / 'data' / f'{s}.csv') for s in protocol['cross_index_features']['indices']]
    target, extra, eligible = build_features(frame, obs, panels)
    assert np.isfinite(target).all() and np.isfinite(extra[eligible]).all()
    obs['panel_valid'] = eligible
    obs.to_csv(OUT / 'observation_table.csv', index=False)
    np.savez_compressed(OUT / 'features.npz', target=target, cross=np.c_[target, extra], eligible=eligible)
    save_json(OUT / 'feature_names.json', feature_names())
    old = pd.read_csv(V2 / 'results/predictions.csv')
    original = old[old.method == 'f_multi_exp_quarter'].sort_values('anchor')
    dev = obs.anchor.isin(original.anchor).to_numpy()
    assert dev.sum() == 272 and eligible[dev].all()
    validation = ((obs.date >= '2018-01-01') & (obs.date <= '2020-12-31') & (obs.completed <= '2020-12-31')).to_numpy()
    assert eligible[validation & (obs.weekday == 4)].all()
    y = obs.exec_return.to_numpy()
    folds, valrows = [], []
    options = [('ridge', str(l), l) for l in protocol['ridge_lambdas']]
    options += [('hgb', str(i), setting) for i, setting in enumerate(protocol['hgb_grid'])]
    for cutoff, end in periods(2018, 2020):
        tr, te = masks(obs, cutoff, end)
        te &= validation
        if not te.any():
            continue
        path = ridge_path(target[tr], y[tr], target[te], protocol['ridge_lambdas'])
        for kind, key, setting in options:
            prediction = path[key] if kind == 'ridge' else hgb_predict(target[tr], y[tr], target[te], setting)
            for ix, value in zip(np.flatnonzero(te), prediction):
                valrows.append(dict(option=key, **prediction_row(obs.iloc[ix], kind + '_target', value, cutoff, tr.sum())))
        folds.append(dict(phase='validation_grid', method='target_all_candidates', cutoff=cutoff, end=end,
                          train_n=int(tr.sum()), first_train=obs.date[tr].min(), last_label=obs.completed[tr].max(),
                          first_signal=obs.date[te].min(), last_signal=obs.date[te].max(), test_n=int(te.sum()), matched=False))
        print(f'Validation {cutoff}: {tr.sum()} train / {te.sum()} weekly; {time.time()-started:.1f}s', flush=True)
    validation_frame = pd.DataFrame(valrows)
    choices, ranking = {}, []
    for kind in ['ridge', 'hgb']:
        scores = []
        for key, g in validation_frame[validation_frame.method == kind + '_target'].groupby('option'):
            mse = float(np.mean((g.predicted_return - g.actual) ** 2))
            setting = float(key) if kind == 'ridge' else protocol['hgb_grid'][int(key)]
            tie = -float(key) if kind == 'ridge' else int(key)
            scores.append((mse, tie, key, setting))
            ranking.append(dict(model=kind, option=key, n=len(g), rmse=np.sqrt(mse), setting=setting))
        best = sorted(scores, key=lambda x: (x[0], x[1]))[0]
        choices[kind] = dict(option=best[2], setting=best[3], validation_rmse=np.sqrt(best[0]))
    save_json(OUT / 'selection.json', dict(frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                                         parameters=choices, rankings=ranking, rule=protocol['selection']))
    validation_frame.to_csv(OUT / 'validation_grid_predictions.csv', index=False)
    print(f'FROZEN PARAMETERS {choices}', flush=True)
    records, selected_val = [], []
    for phase, years, allowed in [('validation', (2018, 2020), validation), ('development', (2021, 2026), dev)]:
        for cutoff, end in periods(*years):
            for kind in ['ridge', 'hgb']:
                for variant in ['target', 'target_matched', 'cross']:
                    name = f'{kind}_{variant}'
                    tr, te = masks(obs, cutoff, end, eligible if variant != 'target' else None)
                    te &= allowed
                    if not te.any():
                        continue
                    x = target if variant != 'cross' else np.c_[target, extra]
                    setting = choices[kind]['setting']
                    if kind == 'ridge':
                        prediction = ridge_path(x[tr], y[tr], x[te], [setting])[str(setting)]
                    else:
                        prediction = hgb_predict(x[tr], y[tr], x[te], setting)
                    for ix, value in zip(np.flatnonzero(te), prediction):
                        row = prediction_row(obs.iloc[ix], name, value, cutoff, tr.sum())
                        (records if phase == 'development' else selected_val).append(row)
                    folds.append(dict(phase=phase, method=name, cutoff=cutoff, end=end,
                                      train_n=int(tr.sum()), first_train=obs.date[tr].min(), last_label=obs.completed[tr].max(),
                                      first_signal=obs.date[te].min(), last_signal=obs.date[te].max(), test_n=int(te.sum()),
                                      matched=variant != 'target', feature_count=x.shape[1]))
                    if phase == 'development' and name == 'ridge_target':
                        for ix in np.flatnonzero(te):
                            records.append(prediction_row(obs.iloc[ix], 'training_mean', y[tr].mean(), cutoff, tr.sum()))
            print(f'{phase} {cutoff} all six arms complete; {time.time()-started:.1f}s', flush=True)
    template = pd.DataFrame(records).query("method == 'ridge_target'").copy().sort_values('anchor')
    for name, value in [('zero_return_cash', 0.), ('buy_hold', 1e-9)]:
        g = template.copy()
        g['method'], g['predicted_return'], g['position'] = name, value, int(value > 0)
        g['correct'] = (g.position == (g.actual > 0)).astype(int)
        records.extend(g.to_dict('records'))
    g = template.copy()
    assert original.anchor.tolist() == g.anchor.tolist()
    g['method'] = 'round2_frozen'
    g['predicted_return'] = original.predicted_return.to_numpy()
    g['position'] = original.position.to_numpy()
    g['correct'] = (g.position == (g.actual > 0)).astype(int)
    records.extend(g.to_dict('records'))
    pd.DataFrame(records).to_csv(OUT / 'local_predictions.csv', index=False)
    pd.DataFrame(selected_val).to_csv(OUT / 'validation_selected_predictions.csv', index=False)
    save_json(OUT / 'folds.json', folds)
    assert previous_evidence() == manifest['previous_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.time()-started,
                    previous_preserved=True, panel_first_valid=obs.date[eligible].min())
    save_json(OUT / 'local_run_manifest.json', manifest)
    print(f'Local experiment completed, {len(records)} development predictions.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
