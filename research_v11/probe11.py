"""Loss and parameter-gradient audits of the eighteen frozen states, training only."""
from common11 import *


def main():
    legacy.initialize()
    check_frozen()
    assert not (OUT / 'probe_manifest.json').exists(), 'Preserve previous probes'
    start = time.time()
    run = manifest('frozen_state_probes')
    save(OUT / 'probe_manifest.json', run)
    directory = OUT / 'training_predictions'
    directory.mkdir(exist_ok=True)
    gradients = OUT / 'gradient_vectors'
    gradients.mkdir(exist_ok=True)
    losses, gradient_rows = [], []
    refs = read(OUT / 'archived_models.json')
    for number, ref in enumerate(refs, 1):
        model, state = load_model(ref)
        values, labels, tr, scales = load_training(ref['cutoff'])
        before = state_hash(model)
        layout = gradient_layout(model)
        if number == 1:
            save(OUT / 'parameter_layout.json', layout)
        else:
            assert layout == read(OUT / 'parameter_layout.json')
        result, prediction = loss_audit(model, values, labels)
        losses.extend(dict(**metadata(ref), **r) for r in result)
        np.savez_compressed(directory / (reference_id(ref) + '.npz'), row_index=tr, **prediction)
        returns, auxiliary = [], []
        for draw in range(-1, cfg()['gradient_audit']['dropout_passes']):
            stochastic = draw >= 0
            seed = cfg()['frozen_state_loss_audit']['dropout_seed_base'] + draw if stochastic else None
            gr, ga = gradient_pass(model, values, labels, stochastic, seed)
            returns.append(gr)
            auxiliary.append(ga)
            gradient_rows.extend(dict(**metadata(ref), mode='dropout' if stochastic else 'eval', draw=draw, **r)
                                 for r in gradient_stats(gr, ga, layout))
        gradient_rows.extend(dict(**metadata(ref), mode='dropout_mean4', draw=-2, **r)
                             for r in gradient_stats(np.mean(returns[1:], axis=0), np.mean(auxiliary[1:], axis=0), layout))
        np.savez_compressed(gradients / (reference_id(ref) + '.npz'), returns=np.stack(returns), auxiliary=np.stack(auxiliary))
        assert before == state_hash(model)
        pd.DataFrame(losses).to_csv(OUT / 'archived_loss_draws.csv', index=False)
        pd.DataFrame(gradient_rows).to_csv(OUT / 'gradient_diagnostics.csv', index=False)
        print(f'Frozen training probes {number}/18; {reference_id(ref)}; {time.time()-start:.1f}s', flush=True)
        del model, values, labels
        torch.cuda.empty_cache()
    check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), models=18, loss_passes=len(losses),
               gradient_passes=90, gradient_summary_rows=len(gradient_rows), elapsed_seconds=time.time()-start)
    save(OUT / 'probe_manifest.json', run)


if __name__ == '__main__':
    main()
