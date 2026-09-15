"""Eighteen fixed-budget paired restarts, with no validation forward passes."""
from common11 import *


def main():
    legacy.initialize()
    check_frozen()
    assert read(OUT / 'probe_manifest.json').get('finished_utc')
    assert not (OUT / 'training_manifest.json').exists(), 'Preserve previous training'
    start = time.time()
    run = manifest('paired_training_restarts')
    save(OUT / 'training_manifest.json', run)
    directory = OUT / 'restart_checkpoints'
    directory.mkdir(exist_ok=True)
    settings = cfg()['restart']
    curves, monitors, draws, refs = [], [], [], []
    for number, job in enumerate(read(OUT / 'restart_plan.json'), 1):
        source = dict(project_file=job['starting_project_file'], sha256=job['starting_sha256'], seed=job['seed'])
        model, starting = load_model(source)
        values, labels, tr, scales = load_training(job['cutoff'])
        assert scales == starting['scales']
        initial_hash = state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        seed = job['seed'] + settings['rng_seed_offset']
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        n = len(tr)
        stats, _ = loss_pass(model, values, labels)
        monitors.append(dict(**metadata(job), epoch=0, **stats))
        for epoch in range(1, settings['epochs'] + 1):
            model.train()
            order = legacy.epoch_order(n, n, rng)
            chunks = batches(order)
            ret, aux, norms = 0., 0., []
            for positions in chunks:
                r, a, norm = update_step(model, optimizer, values, labels, positions, job['auxiliary_weight'])
                ret += r * len(positions)
                aux += a * len(positions)
                norms.append(norm)
            curves.append(dict(**metadata(job), epoch=epoch, train_n=n, presentations=n, optimizer_steps=len(chunks),
                auxiliary_weight=job['auxiliary_weight'], learning_rate=settings['learning_rate'],
                training_order_sha256=array_hash(tr[order]),
                batch_boundaries_sha256=array_hash(np.cumsum([0]+[len(b) for b in chunks])),
                online_return_huber=ret/n, online_auxiliary_mse=aux/n,
                online_optimized_loss=(ret+job['auxiliary_weight']*aux)/n,
                raw_gradient_norm_mean=float(np.mean(norms)), raw_gradient_norm_max=float(max(norms)),
                gradient_clipping_fraction=float(np.mean(np.asarray(norms) > 1.))))
            if epoch in settings['monitor_epochs']:
                stats, _ = loss_pass(model, values, labels)
                monitors.append(dict(**metadata(job), epoch=epoch, **stats))
        result, prediction = loss_audit(model, values, labels)
        draws.extend(dict(**metadata(job), **r) for r in result)
        np.savez_compressed(OUT / 'training_predictions' / (reference_id(job)+'.npz'), row_index=tr, **prediction)
        path = directory / (reference_id(job) + '_restart20.pt')
        torch.save(dict(state_dict={k:t.detach().cpu() for k,t in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(), numpy_rng_state=rng.bit_generator.state,
            cpu_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
            **job, arm='combined', restart_epoch=20, original_epoch=20, train_n=n, scales=scales,
            initial_state_sha256=initial_hash, final_state_sha256=state_hash(model),
            protocol_sha256=sha(ROOT/'protocol.json'), learning_rate=settings['learning_rate'],
            weight_decay=settings['weight_decay'], training_only=True), path)
        refs.append(dict(**metadata(job), project_file=str(path.relative_to(PROJECT)),
                         file=str(path.relative_to(ROOT)), sha256=sha(path), initial_state_sha256=initial_hash,
                         final_state_sha256=state_hash(model)))
        pd.DataFrame(curves).to_csv(OUT / 'restart_training_curves.csv', index=False)
        pd.DataFrame(monitors).to_csv(OUT / 'restart_training_monitors.csv', index=False)
        pd.DataFrame(draws).to_csv(OUT / 'restart_loss_draws.csv', index=False)
        save(OUT / 'restart_models.json', refs)
        print(f'Paired restarts {number}/18; {reference_id(job)}; {time.time()-start:.1f}s', flush=True)
        del model, optimizer, values, labels
        torch.cuda.empty_cache()
    check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), fits=18, checkpoints=18,
               epochs=len(curves), validation_forward_passes=0, elapsed_seconds=time.time()-start)
    save(OUT / 'training_manifest.json', run)


if __name__ == '__main__':
    main()
