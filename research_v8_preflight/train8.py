"""Two disjoint fit-plan workers; frozen model, objective and update budget."""
from common8 import *
import argparse


def main(worker):
    legacy.initialize(); start = time.time()
    check_frozen()
    settings = cfg()['training']
    destination = OUT / f'worker{worker}_manifest.json'
    assert not destination.exists(), 'Preserve existing run'
    run = manifest(f'worker{worker}'); save(destination, run)
    directory = OUT / f'worker{worker}_checkpoints'; directory.mkdir(exist_ok=True)
    obs = pd.read_csv(OUT / 'observation_table.csv')
    jobs = [j for j in fit_plan() if j['worker'] == worker]
    predictions = []; curves = []; checkpoints = []
    for number, job in enumerate(jobs, 1):
        tr, weights, te, full_n = load_policy(job['cutoff'], job['policy'])
        values = legacy.batch_tensors(tr); testing = legacy.batch_tensors(te)
        labels, scales = targets(tr, weights)
        w = None if np.all(weights == 1) else torch.tensor(weights, dtype=torch.float32, device='cuda')
        model = legacy.make_model(settings['arm'], job['seed'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        rng = np.random.default_rng(job['seed'])
        for epoch in range(1, settings['epochs']+1):
            model.train(); order = legacy.epoch_order(len(tr), full_n, rng); running = 0.
            for lo in range(0, full_n, settings['batch_size']):
                batch = torch.tensor(order[lo:lo+settings['batch_size']], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                output, auxiliary = legacy.predict(model, {k: t[batch] for k, t in values.items()}, True)
                loss = joint_loss(output, auxiliary, labels, batch, w)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step(); running += float(loss.detach())*len(batch)
            row = dict(**job, epoch=epoch, train_n=len(tr), presentations=full_n,
                       optimizer_steps=int(np.ceil(full_n/settings['batch_size'])),
                       training_order_sha256=order_hash(tr[order]), online_joint_loss=running/full_n)
            if epoch == settings['epochs']:
                model.eval()
                with torch.inference_mode():
                    output, _ = legacy.predict(model, testing, False)
                    shifted, _ = legacy.predict(model, {k: torch.roll(t, 1, dims=0) for k, t in testing.items()}, False)
                    train_return = []; train_auxiliary = []
                    for lo in range(0, len(tr), 128):
                        p, a = legacy.predict(model, {k: t[lo:lo+128] for k, t in values.items()}, True)
                        train_return.extend(((p-labels['returns'][lo:lo+128])**2).cpu().numpy().astype(float))
                        train_auxiliary.extend(((a-labels['auxiliary'][lo:lo+128])**2).mean(dim=1).cpu().numpy().astype(float))
                p = output.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
                altered = shifted.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
                for i, estimate, change in zip(te, p, altered):
                    predictions.append(dict(**job, row_index=int(i), date=obs.date.iloc[i], actual=float(obs.exec_return.iloc[i]),
                        predicted_return=float(estimate), reassigned_prediction=float(change), training_mean=scales['returns_mean'],
                        source='new_round8_fit', train_n=len(tr)))
                row.update(final_model_train_return_mse=float(np.average(train_return, weights=weights)),
                           final_model_train_auxiliary_mse=float(np.average(train_auxiliary, weights=weights)),
                           validation_mse=float(np.mean((p-obs.exec_return.iloc[te].to_numpy())**2)))
                path = directory / f'{job["policy"]}_{job["cutoff"]}_{job["seed"]}.pt'
                torch.save(dict(state_dict={k:t.detach().cpu() for k,t in model.state_dict().items()}, **job,
                    epoch=epoch, arm=settings['arm'], train_n=len(tr), full_train_n=full_n, scales=scales,
                    protocol_sha256=sha(ROOT / 'protocol.json')), path)
                checkpoints.append(dict(**job, file=str(path.relative_to(ROOT)), project_file=str(path.relative_to(PROJECT)),
                                        sha256=sha(path), epoch=epoch, source='new_round8_fit'))
            curves.append(row)
        pd.DataFrame(predictions).to_csv(OUT / f'worker{worker}_predictions.csv', index=False)
        pd.DataFrame(curves).to_csv(OUT / f'worker{worker}_training_curves.csv', index=False)
        save(OUT / f'worker{worker}_checkpoint_manifest.json', checkpoints)
        print(f'worker{worker}: {number}/{len(jobs)} fits; {job["policy"]} {job["cutoff"]} seed {job["seed"]}; {time.time()-start:.1f}s', flush=True)
        del model, optimizer, values, testing, labels
        torch.cuda.empty_cache()
    assert old_evidence() == run['old_evidence']; check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.time()-start,
               fits=len(jobs), checkpoints=len(checkpoints), predictions=len(predictions), old_preserved=True)
    save(destination, run)
    print(f'Finished worker{worker}: {len(jobs)} fits.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', type=int, choices=[0,1], required=True)
    main(parser.parse_args().worker)
