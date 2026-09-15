"""Freeze a training-only cache and a bounded study before any new probes or fits."""
from common11 import *


def main():
    assert not (OUT / 'preparation_manifest.json').exists(), 'Preserve frozen preparation'
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    run = manifest('preparation')
    run['old_evidence'] = old_evidence()
    obs = pd.read_csv(V10 / 'results/observation_table.csv')
    scales, rows, locals_ = {}, [], []
    with np.load(V5 / 'cache/packed_raw.npz') as packed, np.load(V5 / 'cache/targets.npz') as targets:
        for cutoff, expected_n in zip(cfg()['cutoffs'], [1678, 1921, 2165]):
            with np.load(V10 / f'cache/masks_{cutoff}.npz') as source:
                tr = source['training'].copy()
            expected = np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy())
            np.testing.assert_array_equal(tr, expected)
            assert len(tr) == expected_n and obs.joint_completed.iloc[tr].le(cutoff).all()
            values = {k: packed[k][tr].copy() for k in ['patches', 'geometry', 'valid']}
            assert values['valid'].all()
            scale = {}
            for key in ['returns', 'auxiliary']:
                y = targets[key][tr].astype(float)
                mean, sd = y.mean(axis=0), np.maximum(y.std(axis=0), 1e-6)
                scale[key + '_mean'] = np.asarray(mean).tolist()
                scale[key + '_sd'] = np.asarray(sd).tolist()
                values[key] = ((y - mean) / sd).astype(np.float32)
            scales[cutoff] = scale
            p = CACHE / f'training_{cutoff}.npz'
            np.savez_compressed(p, **values, row_index=tr)
            locals_.append(p)
            rows.extend(dict(cutoff=cutoff, row_index=int(i), date=obs.date.iloc[i],
                             joint_completed=obs.joint_completed.iloc[i]) for i in tr)
    save(OUT / 'training_scales.json', scales)
    pd.DataFrame(rows).to_csv(OUT / 'training_rows.csv', index=False)
    refs = []
    for rule in ['mse', 'huber']:
        for source in read(V10 / f'results/{rule}_checkpoint_manifest.json'):
            ref = {k: source[k] for k in ['cutoff', 'seed', 'project_file', 'sha256']}
            ref['state'] = 'archived_' + rule
            state = torch.load(PROJECT / ref['project_file'], map_location='cpu', weights_only=True)
            assert state['scales'] == scales[ref['cutoff']]
            assert 'optimizer_state_dict' not in state and 'optimizer' not in state
            assert state['epoch'] == 20
            refs.append(ref)
    assert len(refs) == 18
    save(OUT / 'archived_models.json', refs)
    jobs = []
    for ref in refs:
        if ref['state'] == 'archived_huber':
            for arm in cfg()['restart']['arms']:
                jobs.append(dict(cutoff=ref['cutoff'], seed=ref['seed'], state=arm['name'],
                                 auxiliary_weight=arm['auxiliary_weight'],
                                 starting_project_file=ref['project_file'], starting_sha256=ref['sha256']))
    assert len(jobs) == 18
    save(OUT / 'restart_plan.json', jobs)
    locals_ += [OUT / n for n in ['training_scales.json', 'training_rows.csv', 'archived_models.json', 'restart_plan.json']]
    run.update(local_sha256={str(p.relative_to(ROOT)): sha(p) for p in locals_},
               training_counts=[1678, 1921, 2165], cache_has_testing_arrays=False,
               archived_models=18, new_fits=18, completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT / 'preparation_manifest.json', run)
    print(json.dumps(dict(status='FROZEN', training_counts=run['training_counts'], archived_models=18,
                          new_fits=18, old_files=len(run['old_evidence']), protocol_sha256=run['protocol_sha256'])))


if __name__ == '__main__':
    main()
