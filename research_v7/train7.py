"""Independent per-architecture worker for the fixed multiseed deletion study."""
from common7 import *
import argparse


def main(arm):
    legacy.initialize()
    start = time.time()
    settings = cfg()['training']
    destination = OUT / f'{arm}_manifest.json'
    assert not destination.exists(), 'Existing run must be preserved'
    prep = json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))
    tests = json.loads((OUT / 'contract_verification.json').read_text(encoding='utf-8'))
    assert tests['status'] == 'PASS' and tests['source_sha256'] == source_hashes()
    assert prep['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert prep['source_sha256'] == source_hashes() and prep['input_sha256'] == input_hashes()
    for name, digest in prep['local_sha256'].items():
        assert sha(ROOT / name) == digest, name
    run = manifest(arm)
    run['preparation_manifest_sha256'] = sha(OUT / 'preparation_manifest.json')
    save(destination, run)
    directory = OUT / f'{arm}_checkpoints'
    directory.mkdir(exist_ok=True)
    obs = pd.read_csv(OUT / 'observation_table.csv')
    jobs = [j for j in fit_plan() if j['arm'] == arm]
    weight_decay = next(a['weight_decay'] for a in legacy.cfg()['arms'] if a['name'] == arm)
    predictions = []; curves = []; checkpoints = []; baselines = []
    for number, job in enumerate(jobs, 1):
        with np.load(CACHE / f'masks_{job["cutoff"]}.npz') as masks:
            tr = masks[job['history']]; te = masks['testing']; full_count = len(masks['full'])
        values = legacy.batch_tensors(tr); testing = legacy.batch_tensors(te)
        reassigned = {k: torch.roll(v, 1, dims=0) for k, v in testing.items()}
        labels, scales = legacy.standardized_targets(tr)
        ymean = scales['returns_mean']; ysd = scales['returns_sd']
        network = legacy.make_model(arm, job['seed'])
        optimizer = torch.optim.AdamW(network.parameters(), lr=settings['learning_rate'], weight_decay=weight_decay)
        rng = np.random.default_rng(job['seed'])
        for epoch in range(1, settings['epochs'] + 1):
            network.train()
            order = legacy.epoch_order(len(tr), full_count, rng)
            running = 0.
            for lo in range(0, full_count, settings['batch_size']):
                batch = torch.tensor(order[lo:lo+settings['batch_size']], device='cuda')
                v = {k: t[batch] for k, t in values.items()}
                optimizer.zero_grad(set_to_none=True)
                output, auxiliary = legacy.predict(network, v, True)
                loss = ((output-labels['returns'][batch])**2).mean()
                loss = loss + settings['auxiliary_weight']*((auxiliary-labels['auxiliary'][batch])**2).mean()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                running += float(loss.detach())*len(batch)
            row = dict(**job, epoch=epoch, train_n=len(tr), presentations=full_count,
                       optimizer_steps=int(np.ceil(full_count/settings['batch_size'])),
                       training_order_sha256=order_hash(tr[order]), online_joint_loss=running/full_count)
            if epoch in settings['checkpoints']:
                network.eval()
                with torch.inference_mode():
                    out, _ = legacy.predict(network, testing, False)
                    shift, _ = legacy.predict(network, reassigned, False)
                p = out.cpu().numpy().astype(float)*ysd+ymean
                p_shift = shift.cpu().numpy().astype(float)*ysd+ymean
                for i, estimate, altered in zip(te, p, p_shift):
                    predictions.append(dict(**job, epoch=epoch, role=role(arm, epoch), train_n=len(tr),
                                            row_index=int(i), date=obs.date.iloc[i], actual=float(obs.exec_return.iloc[i]),
                                            predicted_return=float(estimate), reassigned_prediction=float(altered),
                                            source='new_round7_fit'))
                train_mse, aux_mse = legacy.evaluate_training(network, values, labels)
                row.update(final_model_train_return_mse=train_mse, final_model_train_auxiliary_mse=aux_mse,
                           validation_mse=float(np.mean((p-obs.exec_return.iloc[te].to_numpy())**2)))
                path = directory / f'{job["history"]}_{job["cutoff"]}_{job["seed"]}_epoch{epoch}.pt'
                torch.save(dict(state_dict={k: t.detach().cpu() for k, t in network.state_dict().items()},
                                **job, epoch=epoch, train_n=len(tr), full_train_n=full_count, scales=scales,
                                protocol_sha256=sha(ROOT / 'protocol.json')), path)
                checkpoints.append(dict(**job, epoch=epoch, role=role(arm, epoch), file=str(path.relative_to(ROOT)),
                                        project_file=str(path.relative_to(PROJECT)), sha256=sha(path), source='new_round7_fit'))
            curves.append(row)
        for i in te:
            baselines.append(dict(history=job['history'], cutoff=job['cutoff'], date=obs.date.iloc[i],
                                  actual=float(obs.exec_return.iloc[i]), train_n=len(tr), training_mean=ymean, zero_return=0.))
        pd.DataFrame(predictions).to_csv(OUT / f'{arm}_predictions.csv', index=False)
        pd.DataFrame(curves).to_csv(OUT / f'{arm}_training_curves.csv', index=False)
        save(OUT / f'{arm}_checkpoint_manifest.json', checkpoints)
        print(f'{arm}: {number}/27 fits; {job["cutoff"]} {job["history"]} seed {job["seed"]}; {time.time()-start:.1f}s', flush=True)
        del network, optimizer, values, testing, reassigned, labels
        torch.cuda.empty_cache()
    pd.DataFrame(baselines).drop_duplicates(['history', 'date']).to_csv(OUT / f'{arm}_baselines.csv', index=False)
    assert old_evidence() == run['old_evidence']
    assert source_hashes() == run['source_sha256'] and input_hashes() == run['input_sha256']
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.time()-start,
               fits=len(jobs), checkpoints=len(checkpoints), predictions=len(predictions), old_preserved=True)
    save(destination, run)
    print(f'Finished {arm}: {len(jobs)} fits, {len(checkpoints)} checkpoints.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['baseline', 'combined'], required=True)
    main(parser.parse_args().arm)
