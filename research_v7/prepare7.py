"""Freeze the bounded replication plan, old full-history forecasts and masks."""
from common7 import *


def main():
    legacy.initialize()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    assert not (OUT / 'preparation_manifest.json').exists(), 'Preparation already frozen'
    run = manifest('preparation')
    (OUT / 'observation_table.csv').write_bytes((V6 / 'results/observation_table.csv').read_bytes())
    for source in (V6 / 'cache').glob('masks_*.npz'):
        (CACHE / source.name).write_bytes(source.read_bytes())
    (OUT / 'fold_history_definitions.csv').write_bytes((V6 / 'results/fold_history_definitions.csv').read_bytes())
    plan = fit_plan()
    assert len(plan) == 54
    save(OUT / 'fit_plan.json', plan)
    previous = pd.read_csv(V6 / 'results/main_seed_predictions.csv')
    reused = previous[previous.arm.isin(['baseline', 'combined']) & previous.epoch.isin([10, 20])].copy()
    reused['role'] = [role(a, e) for a, e in zip(reused.arm, reused.epoch)]
    reused['source'] = 'reused_round6_full'
    assert len(reused) == 1692
    refs = json.loads((V6 / 'results/main_checkpoint_manifest.json').read_text(encoding='utf-8'))
    selected = []
    for r in refs:
        if r['arm'] in ['baseline', 'combined'] and r['epoch'] in [10, 20]:
            selected.append(dict(**r, project_file=str((V6 / r['file']).relative_to(PROJECT)),
                                 role=role(r['arm'], r['epoch']), source='reused_round6_full'))
    assert len(selected) == 36
    save(OUT / 'reused_checkpoint_manifest.json', selected)
    maximum = 0.
    for r in selected:
        keep = ((reused.arm == r['arm']) & (reused.epoch == r['epoch']) &
                (reused.seed == r['seed']) & (reused.cutoff == r['cutoff']))
        reference = reused[keep]
        values = legacy.batch_tensors(reference.row_index.to_numpy())
        saved = torch.load(PROJECT / r['project_file'], map_location='cpu', weights_only=True)
        network = legacy.make_model(r['arm'], r['seed']).eval()
        network.load_state_dict(saved['state_dict'])
        with torch.inference_mode():
            out, _ = legacy.predict(network, values, False)
            shifted, _ = legacy.predict(network, {k: torch.roll(v, 1, dims=0) for k, v in values.items()}, False)
        p = out.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
        np.testing.assert_allclose(p, reference.predicted_return, rtol=0, atol=1e-10)
        maximum = max(maximum, float(np.max(np.abs(p-reference.predicted_return))))
        reused.loc[keep, 'reassigned_prediction'] = shifted.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
        del network, values
        torch.cuda.empty_cache()
    reused.to_csv(OUT / 'reused_full_predictions.csv', index=False)
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), planned_fits=len(plan),
               reused_checkpoints=len(selected), reused_predictions=len(reused), reused_maximum_replay_error=maximum,
               local_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                   list(CACHE.glob('*.npz'))+[OUT / n for n in ['observation_table.csv', 'fold_history_definitions.csv',
                       'fit_plan.json', 'reused_full_predictions.csv', 'reused_checkpoint_manifest.json']]})
    save(OUT / 'preparation_manifest.json', run)
    print(json.dumps(dict(planned_fits=54, new_checkpoints=108, reused_checkpoints=36,
                          reused_predictions=len(reused), previous_evidence_files=len(run['old_evidence'])), indent=2))


if __name__ == '__main__':
    main()
