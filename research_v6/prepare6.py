"""Freeze reused inputs, chronological masks and evidence before any round-6 fit."""
from models6 import *


def main():
    initialize()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    assert not (OUT / 'preparation_manifest.json').exists(), 'Preparation already frozen'
    manifest = start_manifest('preparation')
    obs = pd.read_csv(V5 / 'results/observation_table.csv')
    assert len(obs) == 3780 and obs.date.is_monotonic_increasing
    (OUT / 'observation_table.csv').write_bytes((V5 / 'results/observation_table.csv').read_bytes())
    info = []
    for fold in cfg()['folds']:
        histories, te = fold_indices(obs, fold)
        np.savez_compressed(CACHE / f'masks_{fold["cutoff"]}.npz', **histories, testing=te)
        for name, tr in histories.items():
            removed = np.setdiff1d(histories['full'], tr, assume_unique=True)
            assert obs.joint_completed.iloc[tr].max() <= fold['cutoff'] < obs.date.iloc[te].min()
            info.append(dict(**fold, history=name, train_n=len(tr), full_train_n=len(histories['full']),
                             test_n=len(te), last_train_label=obs.joint_completed.iloc[tr].max(),
                             removed_n=len(removed),
                             removed_start=obs.date.iloc[removed].min() if len(removed) else None,
                             removed_end=obs.date.iloc[removed].max() if len(removed) else None,
                             presentations_per_epoch=len(histories['full']),
                             optimizer_steps_per_epoch=int(np.ceil(len(histories['full']) / 128))))
    pd.DataFrame(info).to_csv(OUT / 'fold_history_definitions.csv', index=False)
    parameters = []
    for arm in cfg()['arms']:
        net = make_model(arm['name'], cfg()['training']['seeds'][0])
        parameters.append(dict(**arm, parameter_count=sum(p.numel() for p in net.parameters())))
        del net
    save(OUT / 'model_parameters.json', parameters)
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), observations=len(obs),
                    cache_sha256={str(p.relative_to(ROOT)): sha(p) for p in CACHE.glob('*.npz')},
                    observation_table_sha256=sha(OUT / 'observation_table.csv'))
    save(OUT / 'preparation_manifest.json', manifest)
    print(json.dumps(dict(observations=len(obs), previous_files=len(manifest['old_evidence']), parameters=parameters), indent=2))
    print(pd.DataFrame(info).to_string(index=False))


if __name__ == '__main__':
    main()
