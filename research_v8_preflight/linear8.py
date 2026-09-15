"""Untuned weighted linear control on the identical flattened raw patches."""
from common8 import *


def fit_ridge(x, y, weights, full_n, alpha=10.):
    w = weights * (full_n/weights.sum())
    xm = np.average(x, axis=0, weights=w)
    xs = np.maximum(np.sqrt(np.average((x-xm)**2, axis=0, weights=w)), 1e-6)
    ym = float(np.average(y, weights=w))
    z = (x-xm)/xs
    gram = z.T @ (w[:,None]*z) + alpha*np.eye(z.shape[1])
    rhs = z.T @ (w*(y-ym))
    beta = np.linalg.solve(gram, rhs)
    residual = float(np.max(np.abs(gram @ beta-rhs)))
    assert residual < 1e-7
    return dict(feature_mean=xm, feature_sd=xs, target_mean=np.asarray(ym), coefficients=beta,
                alpha=np.asarray(alpha), normal_equation_max_error=np.asarray(residual))


def main():
    check_frozen(); destination = OUT / 'linear_manifest.json'
    assert not destination.exists(), 'Preserve existing linear fits'
    run = manifest('linear'); save(destination, run)
    directory = OUT / 'linear_models'; directory.mkdir(exist_ok=True)
    with np.load(V5 / 'cache/packed_raw.npz') as data:
        x = data['patches'].reshape(len(data['patches']), -1).astype(float)
    with np.load(V5 / 'cache/targets.npz') as data:
        y = data['returns'].astype(float)
    obs = pd.read_csv(OUT / 'observation_table.csv')
    rows = []; refs = []
    for fold in cfg()['folds']:
        for policy in cfg()['policies']:
            tr, weights, te, full_n = load_policy(fold['cutoff'], policy)
            model = fit_ridge(x[tr], y[tr], weights, full_n)
            p = (x[te]-model['feature_mean'])/model['feature_sd'] @ model['coefficients'] + model['target_mean']
            altered = np.roll(p, 1)
            for i, estimate, change in zip(te, p, altered):
                rows.append(dict(policy=policy, cutoff=fold['cutoff'], date=obs.date.iloc[i], row_index=int(i),
                    actual=float(y[i]), predicted_return=float(estimate), reassigned_prediction=float(change),
                    training_mean=float(model['target_mean']), train_n=len(tr), source='round8_linear_control'))
            path = directory / f'{policy}_{fold["cutoff"]}.npz'
            np.savez_compressed(path, **model)
            refs.append(dict(policy=policy, cutoff=fold['cutoff'], file=str(path.relative_to(ROOT)), sha256=sha(path),
                             normal_equation_max_error=float(model['normal_equation_max_error'])))
    pd.DataFrame(rows).to_csv(OUT / 'linear_predictions.csv', index=False)
    save(OUT / 'linear_checkpoint_manifest.json', refs)
    check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), fits=len(refs), predictions=len(rows))
    save(destination, run)
    print(json.dumps(dict(fits=len(refs), predictions=len(rows), normal_equations_max_error=max(r['normal_equation_max_error'] for r in refs))))


if __name__ == '__main__':
    main()
